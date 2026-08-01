# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ascend-NPU implementation of masked_scatter_backward.

Semantics
---------
masked_scatter_backward(grad_output, mask, sizes) computes the gradient
w.r.t. ``source`` in the masked_scatter forward op:

    forward:  self[mask] = source[0], source[1], ...  (in mask order)
    backward: d(source)[j] = grad_output[i]  where i is the j-th True
              position in mask (j = cumsum(mask)[i] - 1).

In plain terms: stream-compact the gradient values at True mask positions
into a dense array, preserving order; zero-pad the tail to match the
original source size; then reshape.

Performance-critical observations
----------------------------------
We explored ~15 different implementation strategies over this file's
history.  The conclusions that shaped the current design:

1. CANN native (torch.ops.aten) is 5-10× faster than any Triton
   implementation on Ascend — CANN uses hardware scatter-gather
   instructions that Triton cannot emit.  This is a hardware-level
   gap, not an algorithmic one.

2. Within Triton, the dominant cost is scattered writes
   (tl.store(out_ptr + pos, ...) where pos varies per element).
   Contiguous writes are ~3-12× faster.  Everything that happens
   BEFORE the scatter (cumsum, scan) is fast.

3. Kernel-launch overhead on Ascend is ~80 µs per launch — about the
   same as torch's ENTIRE operator execution for small shapes.
   Minimising launch count is critical.

4. Ascend Triton lacks cross-CTA memory ordering primitives (no
   sem="acq_rel" on atomics, no global memory fence).  This forces
   cross-tile data dependencies (tile N needs tile N-1's total) to
   be resolved via a SEPARATE kernel launch, which inherently acts
   as a global fence.

5. Ascend Triton's tl.atomic_add cannot serve as a multi-CTA barrier
   (attempted and failed — the "last CTA" cannot reliably see other
   CTAs' stores).

6. CPU involvement is counter-productive: a single device→host sync
   is already more expensive than the entire NPU computation for
   typical shapes.

Architecture
------------
Three paths selected by N = mask.numel():

  fused (N ≤ 4096):
    Single-CTA kernel reads bool mask → tl.cumsum → zero-inits
    output → scatters gradient values.  One Triton launch, zero
    overhead.  Achieves 1.1× torch for N=289.

  twolevel (4096 < N ≤ 80M):
    Four-phase pipeline:
      1. tile_cumsum:  multi-CTA per-tile int32 cumsum (contiguous writes)
      2. scan:        single-CTA exclusive scan of tile totals
      3. tile_update: multi-CTA: add per-tile offsets to local cumsum
                      → global 1-based prefix sum (contiguous writes)
      4. scatter:     autotuned masked_select_kernel using the prefix sum
                      (reads contiguous, writes scattered — bottleneck)
      5. zero-pad:    _pad_kernel fills tail with zeros (fused with copy)

  triton (N > 80M):
    Count + scan + scatter-write.  Memory-efficient: stores only
    per-block counts (n_blocks ints) instead of the full prefix sum
    (N ints = up to 2.6 GB).  The write kernel recomputes per-CTA
    cumsum internally, trading compute for memory.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants  (design rationale in module docstring above)
# ---------------------------------------------------------------------------

# TILE_SIZE: max elements per CTA for cumsum operations.
# 4096 is the sweet spot: large enough to minimise tile count (and thus
# scan overhead), small enough to avoid register spill in tl.cumsum.
_TILE_SIZE = 4096

# TWOLEVEL_MAX_N: above this, the full prefix_sum buffer (N × 4 bytes)
# would exceed ~320 MB, causing memory pressure.  We fall back to the
# memory-efficient triton path which stores only per-block counts.
_TWOLEVEL_MAX_N = 80 * 1024 * 1024  # 80M

# Block-size bounds for the triton fallback path.
_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 4096

# MAX_SCAN_BLOCK: max elements a single-CTA scan can handle.
# Above this we use a two-level scan (chunk → scan → add-offsets).
_MAX_SCAN_BLOCK = 4096


# =========================================================================
# FUSED PATH  —  N ≤ TILE_SIZE  (single CTA, one kernel launch)
#
# Design: everything in one kernel — no mask copy, no separate zeros,
# no slice copy.  The single-CTA guarantees no cross-CTA races, so
# we can safely zero-init and scatter in the same launch.
# =========================================================================


@libentry()
@triton.jit(do_not_specialize=["N"])
# do_not_specialize: N varies per shape; avoid recompilation for each N.
def _fused_kernel(
    grad_ptr,     # pointer to gradient values (float)
    mask_ptr,     # pointer to boolean mask
    out_ptr,      # pointer to output buffer (pre-allocated, torch.empty)
    N,            # number of elements to process
    out_numel,    # total output size (prod(sizes)); may be > N
    BLOCK_SIZE: tl.constexpr,  # compile-time constant: next_power_of_2(N)
):
    """Single-CTA fused: read bool → cumsum → zero-init → scatter.

    Called with grid=(1,) — exactly one CTA processes all N elements.
    torch.empty is used for the output (no separate zeros kernel).
    """
    # ---- compute per-element thread indices ----
    # tl.arange(0, BLOCK_SIZE) produces [0, 1, ..., BLOCK_SIZE-1].
    # Each thread in the CTA handles one offset.
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask out threads beyond N (last CTA may have fewer valid elements).
    block_mask = offsets < N

    # ---- read inputs ----
    # mask_val: Triton int1 (bool), loaded directly from the bool tensor.
    # No explicit .to(torch.int32) copy needed — Triton handles the load.
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0)
    # mask_int: convert to int32 for tl.cumsum (cumsum on bool would
    # implicitly cast, but explicit is clearer and avoids Ascend
    # compiler quirks with bool arithmetic).
    mask_int = mask_val.to(tl.int32)
    # pos: 0-based exclusive scan of mask.
    # tl.cumsum produces [1,1,2,3,3,4,...] for [T,F,T,T,F,T,...].
    # Subtract 1 → exclusive: [0,×,1,2,×,3,...] (× = don't-care, masked).
    pos = tl.cumsum(mask_int, axis=0) - 1

    # gradient values — only needed at True positions; other values
    # are masked out by the store mask below.
    grad_val = tl.load(grad_ptr + offsets, mask=block_mask, other=0)

    # ---- zero-init the output (single CTA → no cross-CTA race) ----
    # All threads write 0.0 to their assigned output positions.
    # out_numel may be > N when source.numel() > mask.numel(); the
    # zero_mask ensures we only write within allocated bounds.
    tl.store(out_ptr + offsets, 0.0, mask=(offsets < out_numel))

    # ---- scatter gradient at compacted positions ----
    # Four conditions for a valid scatter write:
    #   block_mask   — within this CTA's data range
    #   mask_bool    — mask is True at this position
    #   pos >= 0     — at least one True seen so far (exclusive scan guard)
    #   pos < out_numel — write position within output bounds
    tl.store(
        out_ptr + pos,                          # scattered destination
        grad_val,                               # value to write
        mask=block_mask
             & mask_val.to(tl.int1)             # only True mask positions
             & (pos >= 0)                       # first True → pos=0, valid
             & (pos < out_numel),               # bounds check
    )


# =========================================================================
# PAD KERNEL  —  shared between twolevel and triton paths
#
# Replaces torch.zeros + slice copy with a single Triton kernel,
# using torch.empty for the output buffer.  Saves one kernel launch.
# =========================================================================


@libentry()
@triton.jit(do_not_specialize=["numel"])
def _pad_kernel(
    selected_ptr,  # compacted gradient values
    out_ptr,       # output buffer (torch.empty)
    n_selected,    # number of valid elements in selected_ptr
    numel,         # total output size (prod(sizes))
    BLOCK_SIZE: tl.constexpr,
):
    """Fused zero-pad + copy: out[0:n_selected] = selected, rest = 0.

    Each CTA writes a block of the output.  Positions < n_selected
    receive the compacted gradient; positions >= n_selected receive 0.
    """
    pid = ext.program_id(axis=0)
    # Compute this CTA's global offset range.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < numel                    # valid output positions
    in_range = offsets < n_selected        # positions that have data

    # Load selected values where we have data; other=0 fills the tail.
    val = tl.load(selected_ptr + offsets, mask=(m & in_range), other=0)
    # Store to output covering all positions in this CTA's range.
    tl.store(out_ptr + offsets, val, mask=m)


# =========================================================================
# TWOLEVEL PATH  —  TILE_SIZE < N ≤ 80M
#
# Four-phase pipeline:
#   1. tile_cumsum   — per-tile local cumsum + store tile totals
#   2. scan          — exclusive scan of tile totals → per-tile offsets
#   3. tile_update   — add tile offsets to local cumsum → global prefix
#   4. scatter       — autotuned scatter-write using global prefix sum
#   5. zero-pad      — _pad_kernel
#
# Why four phases instead of one?  Cross-tile data dependency: tile N
# needs tile 0..N-1's total True count to compute global positions.
# Ascend Triton lacks cross-CTA memory ordering, so this dependency
# MUST be resolved by a kernel-launch boundary (which acts as a
# hardware-level global fence).
# =========================================================================


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tile_cumsum_kernel(
    inp_ptr,          # pointer to bool mask (or int32 after conversion)
    out_ptr,          # pointer to output: local per-tile cumsum (int32)
    tile_totals_ptr,  # pointer to per-tile total True counts (int64)
    N,                # total number of elements
    TILE_SIZE: tl.constexpr,
):
    """Phase 1: per-tile inclusive cumsum + store tile total.

    Each CTA processes one tile of TILE_SIZE elements.
    - Computes tl.cumsum of the boolean mask → int32 cumsum (0,1,2,...)
    - Stores the cumsum values (contiguous writes — fast)
    - Stores the tile's total True count to tile_totals_ptr[pid]

    The cumsum stays in int32: max value per tile = TILE_SIZE = 4096,
    well within int32 range.  No precision issues.

    Accepts bool input directly: tl.load reads as int1, .to(tl.int32)
    converts to 0/1.  Avoids a separate mask→int32 memory copy.
    """
    pid = ext.program_id(axis=0)
    # Compute this tile's offset range.
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    m = offsets < N

    # Load bool mask → int32 (0 or 1).
    # .to(tl.int32) on Ascend: True→1, False→0.  This avoids the
    # bool→float32→int32 path that the recursive cumsum takes and
    # which loses precision for large N.
    x = tl.load(inp_ptr + offsets, mask=m, other=0).to(tl.int32)

    # Inclusive cumsum: [1,1,2,2,3,...] for [1,0,1,0,1,...]
    cumsum = tl.cumsum(x, axis=0)

    # Store to output (contiguous — fast).
    tl.store(out_ptr + offsets, cumsum, mask=m)

    # Store per-tile total (one int64 per CTA).  This is the ONLY
    # cross-CTA data needed by the scan phase.
    tl.store(tile_totals_ptr + pid, tl.sum(x, axis=0))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tile_update_kernel(
    out_ptr,           # pointer to local cumsum values (int32)
    tile_offsets_ptr,  # pointer to per-tile exclusive scan offsets (int64)
    N,                 # total number of elements
    TILE_SIZE: tl.constexpr,
):
    """Phase 3: add tile offsets to local cumsum → global prefix sum.

    Each CTA reads its tile offset from tile_offsets_ptr[pid] (computed
    by the scan in phase 2) and adds it to every local cumsum value in
    its tile.

    This is a trivial contiguous-read + contiguous-write kernel.
    Launch overhead (~80 µs) dominates; actual compute is ~12 µs.

    Result: out_ptr[i] now holds the 1-based GLOBAL inclusive cumsum,
    i.e., the number of True mask elements in [0..i].
    """
    pid = ext.program_id(axis=0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    m = offsets < N

    # Read local cumsum value.
    val = tl.load(out_ptr + offsets, mask=m, other=0)
    # Read per-tile offset (one value per CTA, broadcast to all threads).
    # Add to produce 1-based global cumsum.
    tl.store(out_ptr + offsets, val + tl.load(tile_offsets_ptr + pid), mask=m)


# ---------------------------------------------------------------------------
# Scan sub-kernels  (used by _exclusive_scan)
#
# The scan is over tile_totals (n_tiles elements, where n_tiles =
# ceil(N / 4096)).  For N ≤ 16M, n_tiles ≤ 4096 — single-CTA suffices.
# For larger N, we use a two-level scan:
#   Level 1: chunk_scan — multiple CTAs, each scanning a chunk
#   Level 2: scan       — single CTA scanning the chunk totals
#   Level 3: add_offsets — apply chunk offsets to chunk-internal results
#
# The two-level design mirrors normed_cumsum's pattern from cumsum.py
# and guarantees fixed O(1) launches regardless of N, unlike the
# recursive scan_then_fan_col which is O(log N).
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _scan_kernel(
    counts_ptr,    # input: array to be scanned
    part_sums_ptr, # output: exclusive scan result
    n_elem,        # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Single-CTA exclusive scan: reads n_elem values, writes result.

    Launched with grid=(1,) — one CTA processes the entire input.
    Used for both the primary scan (when n_tiles ≤ 4096) and the
    inner level of the two-level scan.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    m = offsets < n_elem

    # Load input values.
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    # Inclusive cumsum.
    cumsums = tl.cumsum(counts, axis=0)
    # Exclusive: subtract the current element from the cumsum.
    # [1,3,6] → [0,1,3] for input [1,2,3].
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=m)


@libentry()
@triton.jit
def _chunk_scan_kernel(
    counts_ptr,       # input array
    part_sums_ptr,    # output: per-chunk exclusive scan
    chunk_totals_ptr, # output: per-chunk total (sum of each chunk)
    n_elem,           # total elements in counts_ptr
    CHUNK_SIZE: tl.constexpr,
):
    """Multi-CTA chunked scan — Level 1 of the two-level scan.

    Each CTA processes CHUNK_SIZE elements:
    - Computes local exclusive scan → part_sums_ptr
    - Stores the sum of its chunk → chunk_totals_ptr[pid]

    The chunk totals are then scanned by _scan_kernel (Level 2) to
    produce per-chunk global offsets, which _add_offsets_kernel
    (Level 3) applies to the local scan results.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem

    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=m)

    # Store the sum of this chunk for the next scan level.
    tl.store(chunk_totals_ptr + pid, tl.sum(counts, axis=0))


@libentry()
@triton.jit
def _add_offsets_kernel(
    part_sums_ptr,    # per-chunk exclusive scan results (modified in-place)
    chunk_offsets_ptr, # global offsets for each chunk (from Level 2 scan)
    n_elem,           # total elements
    CHUNK_SIZE: tl.constexpr,
):
    """Level 3 of the two-level scan: add per-chunk offsets.

    Each CTA reads its chunk's global offset and adds it to every
    element in the chunk, converting the local exclusive scan into
    a global exclusive scan.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem

    val = tl.load(part_sums_ptr + offsets, mask=m, other=0)
    # One value per CTA — broadcast to all threads.
    tl.store(part_sums_ptr + offsets,
             val + tl.load(chunk_offsets_ptr + pid), mask=m)


def _exclusive_scan(arr, n_elems, device):
    """Exclusive scan over an array, on-device.

    Dispatch:
    - n_elems ≤ _MAX_SCAN_BLOCK (4096): single-CTA _scan_kernel
    - n_elems > _MAX_SCAN_BLOCK:   two-level (chunk → scan totals → add)

    Returns a DEVICE tensor of shape (n_elems,) with int64 values.
    The caller is responsible for freeing the intermediate tensors
    (they go out of scope when the function returns).
    """
    # Allocate output — int64 to hold cumulative True counts up to N.
    part_sums = torch.empty(n_elems, dtype=torch.int64, device=device)

    if n_elems <= _MAX_SCAN_BLOCK:
        # Single-CTA: fast path for small tile counts.
        scan_block = triton.next_power_of_2(n_elems)
        _scan_kernel[(1,)](arr, part_sums, n_elems, BLOCK_SIZE=scan_block)
    else:
        # ---- Level 1: chunked scan ----
        n_chunks = triton.cdiv(n_elems, _MAX_SCAN_BLOCK)
        chunk_totals = torch.empty(n_chunks, dtype=torch.int64, device=device)
        _chunk_scan_kernel[(n_chunks,)](
            arr, part_sums, chunk_totals, n_elems,
            CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )

        # ---- Level 2: scan the chunk totals ----
        chunk_offsets = torch.empty(n_chunks, dtype=torch.int64, device=device)
        scan_block2 = triton.next_power_of_2(n_chunks)
        _scan_kernel[(1,)](
            chunk_totals, chunk_offsets, n_chunks, BLOCK_SIZE=scan_block2,
        )

        # ---- Level 3: add chunk offsets to local results ----
        _add_offsets_kernel[(n_chunks,)](
            part_sums, chunk_offsets, n_elems, CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )

    return part_sums


def _multi_tile_path(grad_output, mask, numel, N):
    """Twolevel path: tile_cumsum + scan + tile_update + scatter + pad.

    This is the workhorse for the most common shape range (4K < N ≤ 80M).
    Each phase is a separate kernel launch because Ascend Triton lacks
    cross-CTA memory ordering — the kernel-launch boundary provides the
    necessary global memory fence.

    Performance breakdown (N=65536, float32, µs):
      phase 1 (tile_cumsum):  67   contiguous int32 writes
      phase 2 (scan):         89   mostly launch overhead (1 CTA, ~1µs compute)
      phase 3 (tile_update):  92   contiguous int32 read+write
      phase 4 (scatter):     350   scattered float writes ← BOTTLENECK
      phase 5 (pad):          60   contiguous float writes
      TOTAL:                 658
      torch (CANN):          190   (2 launches: hardware cumsum + scatter)

    Alternatives explored and rejected:
    - CPU-side scan (phase 2 on CPU):  .cpu() sync → 18× slower
    - Fused scatter (phase 3+4 combined): register spill → 430 µs for cumsum
    - Chunked per-tile compact (no global scatter): register spill in
      combined cumsum+grad_store kernel → slower
    - Atomic barrier to fuse phases 1-3: tl.atomic_add lacks memory
      ordering on Ascend → last CTA cannot see other CTAs' stores
    - Bit-shift to eliminate phase 3: UB overflow or 2× slower scatter
    """
    device = grad_output.device
    # torch_device_fn.device() ensures all Triton launches target the
    # correct NPU device stream.
    with torch_device_fn.device(device):
        # Flatten the mask to 1D for tile-based processing.
        # .ravel() returns a view (no copy) when the tensor is contiguous.
        mask_flat = mask.ravel()
        n_tiles = triton.cdiv(N, _TILE_SIZE)

        # ---- Phase 1: per-tile local cumsum ----
        # prefix_sum: N int32 values = N × 4 bytes.
        # For N=16M this is 64 MB — manageable.  For N>80M the triton
        # fallback path avoids this allocation.
        prefix_sum = torch.empty(N, dtype=torch.int32, device=device)
        # tile_totals: n_tiles int64 values (one per CTA).
        tile_totals = torch.empty(n_tiles, dtype=torch.int64, device=device)
        _tile_cumsum_kernel[(n_tiles,)](
            mask_flat,               # bool input — kernel converts to int32
            prefix_sum,              # output: local cumsum per tile
            tile_totals,             # output: per-tile True counts
            N, TILE_SIZE=_TILE_SIZE,
        )

        # ---- Phase 2-3: scan + tile_update → global prefix sum ----
        # Phase 2 (scan): exclusive scan of tile_totals → tile_offsets.
        # For n_tiles ≤ 4096: single CTA, ~1 µs compute + ~80 µs launch.
        tile_offsets = _exclusive_scan(tile_totals, n_tiles, device)

        # Phase 3 (tile_update): tile_offsets[pid] + prefix_sum[i] → global.
        # Contiguous read+write — fast (~12 µs compute).
        _tile_update_kernel[(n_tiles,)](
            prefix_sum, tile_offsets, N, TILE_SIZE=_TILE_SIZE,
        )

        # ---- Phase 4: autotuned scatter-write ----
        # prefix_sum[-1] is the TOTAL number of True mask elements
        # (last cumsum value = mask.sum()).  .item() triggers a
        # device→host sync, but it's unavoidable — we need this value
        # to allocate the compacted output tensor.
        n_selected = prefix_sum[-1].item()
        mask_selected = torch.empty(
            n_selected, dtype=grad_output.dtype, device=device,
        )

        # masked_select_kernel is the Ascend-platform autotuned scatter
        # kernel.  It is NOT our code — it's provided by the Ascend
        # masked_select operator.  We import it lazily to avoid
        # circular import issues.
        #
        # The kernel reads prefix_sum[i], mask[i], grad[i] (all
        # contiguous), computes pos = prefix_sum[i] - 1, and does a
        # scattered store: out[pos] = grad[i] (where mask[i] is True).
        # The scattered store is the performance bottleneck.
        #
        # Autotune configs: BLOCK_SIZE ∈ {512, 1024, 2048, 4096}
        # tuned by n_elements.  The autotuner selects the best
        # BLOCK_SIZE at first call; subsequent calls use the cached
        # value.
        from .masked_select import masked_select_kernel

        # The grid lambda computes the number of CTAs from the
        # autotuned BLOCK_SIZE.  Smaller BLOCK_SIZE → more CTAs →
        # better parallelism but more scattered-write transactions.
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        masked_select_kernel[grid](
            grad_output.ravel(),         # gradient values (float)
            mask_flat.to(torch.int32),   # mask (int32 — avoids bool→float32)
            prefix_sum,                  # 1-based global cumsum
            mask_selected,               # output: compacted gradient values
            N,                           # total elements
        )

    # ---- Phase 5: zero-pad to output size ----
    # When mask.sum() < prod(sizes) (typical: ~60% True), the trailing
    # elements of the output must be zero.  _pad_kernel fuses the copy
    # and zero-fill into one launch, using torch.empty to avoid a
    # separate torch.zeros kernel.
    if n_selected < numel:
        out = torch.empty(numel, dtype=mask_selected.dtype, device=device)
        _pad_kernel[(triton.cdiv(numel, _TILE_SIZE),)](
            mask_selected, out, n_selected, numel, BLOCK_SIZE=_TILE_SIZE,
        )
        return out
    return mask_selected


# =========================================================================
# TRITON FALLBACK PATH  —  N > 80M
#
# For very large N, the twolevel path's prefix_sum allocation would
# exceed 320 MB (80M × 4 bytes).  This path stores only per-block
# counts (n_blocks × 8 bytes = ~160 KB for N=655M), trading compute
# (per-CTA tl.cumsum in the write kernel) for memory.
#
# The write kernel's per-CTA cumsum + scattered writes are slower
# than the twolevel scatter, but for N > 80M the memory savings
# outweigh the compute cost.
# =========================================================================


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tr_count_kernel(
    mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    """Count True mask elements per block — one int64 per CTA.

    Identical in spirit to _tile_cumsum_kernel but stores only the
    PER-BLOCK SUM (one value per CTA), not the full cumsum array.
    This is what makes the path memory-efficient.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load bool mask, convert to int32, sum → one value per CTA.
    tl.store(
        counts_ptr + pid,
        tl.sum(
            tl.load(mask_ptr + offsets, mask=offsets < N, other=0).to(tl.int32),
            axis=0,
        ),
    )


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tr_write_kernel(
    grad_ptr,      # gradient values
    mask_ptr,      # boolean mask
    part_sums_ptr, # per-block exclusive scan offsets
    out_ptr,       # pre-zeroed output buffer
    N,             # total elements
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter-write with per-CTA cumsum recomputation.

    Each CTA:
    1. Reads its global offset from part_sums_ptr[pid]
    2. Reads mask and gradient for its block
    3. Recomputes the intra-block exclusive cumsum (tl.cumsum)
    4. Scatters gradient values to output[global_offset + local_pos]

    The tl.cumsum recomputation is the cost we pay for avoiding the
    full prefix_sum allocation.  For N > 80M, the memory savings
    (~320 MB) outweigh the extra compute.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N

    # Read mask and gradient for this block.
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    grad_val = tl.load(grad_ptr + offsets, mask=block_mask, other=0)

    # Global offset: number of True elements in all previous blocks.
    global_offset = tl.load(part_sums_ptr + pid)

    # Recompute intra-block exclusive cumsum.
    # This is the same computation that _tile_cumsum_kernel did for the
    # twolevel path, but done per-block in the write kernel to avoid
    # storing the intermediate prefix_sum array.
    local_pos = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1

    # Scattered store — same bottleneck as twolevel scatter.
    tl.store(out_ptr + global_offset + local_pos, grad_val,
             mask=(block_mask & mask_val))


def _triton_fallback(grad_output, mask, numel, N):
    """Memory-efficient path for N > 80M.

    1. Count per block (stores only n_blocks ints)
    2. Exclusive scan of block counts → block offsets
    3. Write kernel with recomputed per-CTA cumsum

    The output is pre-zeroed via torch.zeros (CANN native when
    use_gems excludes zero_, or flag_gems zeros Triton kernel).
    For very large N, torch.zeros is a single kernel launch and
    its overhead is amortized.
    """
    # BLOCK_SIZE: clamped to [128, 4096] via bracket_next_power_of_2.
    BLOCK_SIZE = bracket_next_power_of_2(N, _MIN_BLOCK_SIZE, _MAX_BLOCK_SIZE)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    device = grad_output.device

    # Pre-zero the output — one full sweep; for N > 80M this is
    # dominated by the write kernel's scattered stores.
    out = torch.zeros(numel, dtype=grad_output.dtype, device=device)

    with torch_device_fn.device(device):
        # ---- Phase 1: per-block count ----
        # counts: n_blocks × 8 bytes.  For N=655M with BLOCK_SIZE=4096:
        # n_blocks = 160K, counts = 1.25 MB — negligible.
        counts = torch.empty(n_blocks, dtype=torch.int64, device=device)
        _tr_count_kernel[(n_blocks,)](
            mask.ravel(), counts, N, BLOCK_SIZE=BLOCK_SIZE,
        )

        # ---- Phase 2: exclusive scan of block counts ----
        # Reuses the same _exclusive_scan as the twolevel path.
        part_sums = _exclusive_scan(counts, n_blocks, device)

        # ---- Phase 3: scatter-write (with per-CTA cumsum) ----
        _tr_write_kernel[(n_blocks,)](
            grad_output.ravel(), mask.ravel(), part_sums, out, N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


# =========================================================================
# PUBLIC ENTRY POINT
# =========================================================================


def masked_scatter_backward(grad_output, mask, sizes):
    """Backward of masked_scatter with respect to ``source``.

    Registered as the Ascend backend implementation of
    aten::masked_scatter_backward.

    Parameters
    ----------
    grad_output : Tensor
        Gradient of the loss w.r.t. the output of masked_scatter.
        Same shape as the ``self`` tensor in the forward pass.
    mask : Tensor (bool)
        The mask tensor from the forward pass.  Same shape as grad_output
        (after broadcasting).
    sizes : list[int]
        Shape of the ``source`` tensor from the forward pass.

    Returns
    -------
    Tensor
        Gradient w.r.t. ``source``, shaped as ``sizes``.

    Dispatch
    --------
    N ≤ 4096:
        _fused_kernel — single-CTA Triton kernel (1 launch).
        Fastest path, zero overhead.

    4096 < N ≤ 80M:
        _multi_tile_path — tile_cumsum + scan + tile_update + autotuned
        scatter (4-5 launches).  The dominant path for typical shapes.

    N > 80M:
        _triton_fallback — count + scan + scatter-write (3-4 launches).
        Memory-efficient; avoids allocating the full prefix_sum buffer.
    """
    logger.debug("GEMS_ASCEND MASKED_SCATTER_BACKWARD")

    # Compute total output size: prod(sizes).
    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()

    if N <= _TILE_SIZE:
        # —— FUSED: single CTA, 1 kernel launch ——
        # BLOCK_SIZE = next power of 2: e.g., N=289 → 512, N=4096 → 4096.
        BLOCK_SIZE = triton.next_power_of_2(N)
        # torch.empty avoids a separate zeros kernel; _fused_kernel
        # zero-inits the output internally.
        out = torch.empty(numel, dtype=grad_output.dtype,
                          device=grad_output.device)
        with torch_device_fn.device(grad_output.device):
            _fused_kernel[(1,)](
                grad_output.ravel(), mask.ravel(), out,
                N, numel, BLOCK_SIZE=BLOCK_SIZE,
            )
    elif N <= _TWOLEVEL_MAX_N:
        # —— TWOLEVEL: multi-CTA pipeline, 4-5 launches ——
        out = _multi_tile_path(grad_output, mask, numel, N)
    else:
        # —— TRITON FALLBACK: memory-efficient, 3-4 launches ——
        out = _triton_fallback(grad_output, mask, numel, N)

    # The result is always 1-D compacted + padded; reshape to target.
    return out.view(sizes)
