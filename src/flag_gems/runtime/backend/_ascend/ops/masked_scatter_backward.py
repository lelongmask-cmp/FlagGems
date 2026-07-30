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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ascend masked_scatter_backward — normed_cumsum-style two-level prefix sum.
#
# Instead of relying on the recursive scan_then_fan_col cumsum (O(log N)
# kernel launches, grid-limited), we compute the 1-based prefix sum over
# the boolean mask ourselves using a fixed-launch two-level design:
#
#   1. tile_cumsum:  split N into tiles, each CTA computes local cumsum
#                    and stores the tile total.
#   2. scan totals:  exclusive scan over tile totals (single-CTA or
#                    two-level for > _MAX_SCAN_BLOCK tiles).
#   3. tile_update:  add per-tile offsets to local cumsum values.
#   4. scatter-write: autotuned masked_select_kernel with prefix sum.
#   5. zero-pad:     pre-allocation + slice copy.
#
# This gives a fixed 5-7 kernel launches for any N (vs. recursive cumsum
# which adds 2 launches per recursion level).  No torch compute ops.
# ---------------------------------------------------------------------------

_MAX_SCAN_BLOCK = 4096
_TILE_SIZE = 4096
# For N > 100M, use a larger tile to keep n_tiles manageable.
_TILE_SIZE_HUGE = 16384


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tile_cumsum_kernel(
    inp_ptr,
    out_ptr,
    tile_totals_ptr,
    N,
    TILE_SIZE: tl.constexpr,
):
    """Per-tile inclusive cumsum + store tile total.

    Each CTA processes one tile of up to TILE_SIZE elements.
    For int32 input (mask values 0/1), the cumsum stays in exact int32
    arithmetic (no float32 precision loss).
    """
    pid = tl.program_id(0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offsets < N

    x = tl.load(inp_ptr + offsets, mask=mask, other=0)
    # int32 input → int32 cumsum (exact)
    cumsum = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offsets, cumsum, mask=mask)

    total = tl.sum(x, axis=0)
    tl.store(tile_totals_ptr + pid, total)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tile_update_kernel(
    out_ptr,
    tile_offsets_ptr,
    N,
    TILE_SIZE: tl.constexpr,
):
    """Add per-tile global offsets to each tile's local cumsum."""
    pid = tl.program_id(0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offsets < N

    val = tl.load(out_ptr + offsets, mask=mask, other=0)
    offset = tl.load(tile_offsets_ptr + pid)
    tl.store(out_ptr + offsets, val + offset, mask=mask)


@libentry()
@triton.jit
def _scan_kernel(
    counts_ptr,
    part_sums_ptr,
    n_elem,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-CTA exclusive scan."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=mask)


@libentry()
@triton.jit
def _chunk_scan_kernel(
    counts_ptr,
    part_sums_ptr,
    chunk_totals_ptr,
    n_elem,
    CHUNK_SIZE: tl.constexpr,
):
    """Multi-CTA chunked exclusive scan + store chunk totals."""
    pid = tl.program_id(0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=mask)
    total = tl.sum(counts, axis=0)
    tl.store(chunk_totals_ptr + pid, total)


@libentry()
@triton.jit
def _add_offsets_kernel(
    part_sums_ptr,
    chunk_offsets_ptr,
    n_elem,
    CHUNK_SIZE: tl.constexpr,
):
    """Add per-chunk offsets to local exclusive scan results."""
    pid = tl.program_id(0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = offsets < n_elem
    val = tl.load(part_sums_ptr + offsets, mask=mask, other=0)
    chunk_offset = tl.load(chunk_offsets_ptr + pid)
    tl.store(part_sums_ptr + offsets, val + chunk_offset, mask=mask)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _exclusive_scan(arr, n_elems, device):
    """Exclusive scan over an array, on-device.

    Single-CTA for n_elems ≤ _MAX_SCAN_BLOCK, two-level otherwise.
    """
    part_sums = torch.empty(n_elems, dtype=torch.int64, device=device)

    if n_elems <= _MAX_SCAN_BLOCK:
        scan_block = triton.next_power_of_2(n_elems)
        _scan_kernel[(1,)](arr, part_sums, n_elems, BLOCK_SIZE=scan_block)
    else:
        n_chunks = triton.cdiv(n_elems, _MAX_SCAN_BLOCK)
        chunk_totals = torch.empty(n_chunks, dtype=torch.int64, device=device)
        _chunk_scan_kernel[(n_chunks,)](
            arr, part_sums, chunk_totals, n_elems,
            CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )
        chunk_offsets = torch.empty(n_chunks, dtype=torch.int64, device=device)
        scan_block2 = triton.next_power_of_2(n_chunks)
        _scan_kernel[(1,)](
            chunk_totals, chunk_offsets, n_chunks, BLOCK_SIZE=scan_block2,
        )
        _add_offsets_kernel[(n_chunks,)](
            part_sums, chunk_offsets, n_elems, CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )
    return part_sums


def _two_level_prefix_sum(mask_flat, N, device):
    """Compute 1-based inclusive prefix sum of a 1-D int32 mask.

    Uses the normed_cumsum two-level pattern:
      1. tile_cumsum   — local cumsum per tile + store tile total
      2. exclusive scan — over tile totals (single-CTA or two-level)
      3. tile_update   — add per-tile global offsets

    Returns a 1-D int32 tensor of shape (N,) with the prefix sum.
    Always 3-5 kernel launches regardless of N.
    """
    tile_size = _TILE_SIZE_HUGE if N > 100_000_000 else _TILE_SIZE
    n_tiles = triton.cdiv(N, tile_size)

    # 1. per-tile cumsum
    prefix_sum = torch.empty(N, dtype=torch.int32, device=device)
    tile_totals = torch.empty(n_tiles, dtype=torch.int64, device=device)
    _tile_cumsum_kernel[(n_tiles,)](
        mask_flat, prefix_sum, tile_totals, N,
        TILE_SIZE=tile_size,
    )

    # 2. exclusive scan of tile totals → per-tile offsets
    tile_offsets = _exclusive_scan(tile_totals, n_tiles, device)

    # 3. add tile offsets to local cumsum values
    _tile_update_kernel[(n_tiles,)](
        prefix_sum, tile_offsets, N, TILE_SIZE=tile_size,
    )

    return prefix_sum


# ---------------------------------------------------------------------------
# Main implementation
# ---------------------------------------------------------------------------


def masked_scatter_backward(grad_output, mask, sizes):
    """Backward of masked_scatter w.r.t. ``source``.

    Uses a normed_cumsum-style two-level prefix sum (fixed O(1) kernel
    launches) + the autotuned Ascend masked_select scatter-write kernel.
    No recursive cumsum, no grid-limit ceiling.
    """
    logger.debug("GEMS_ASCEND MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()
    device = grad_output.device

    with torch_device_fn.device(device):
        # 1. two-level prefix sum (int32 for exact arithmetic)
        mask_flat = mask.ravel().to(torch.int32)
        prefix_sum = _two_level_prefix_sum(mask_flat, N, device)

        # 2. allocate compacted output
        n_selected = prefix_sum[-1].item()
        mask_selected = torch.empty(
            n_selected, dtype=grad_output.dtype, device=device
        )

        # 3. autotuned scatter-write (same kernel as Ascend masked_select)
        from .masked_select import masked_select_kernel

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        masked_select_kernel[grid](
            grad_output.ravel(),
            mask_flat,
            prefix_sum,
            mask_selected,
            N,
        )

        # 4. zero-pad to target size
        if n_selected < numel:
            out = torch.zeros(
                numel, dtype=mask_selected.dtype, device=device
            )
            out[:n_selected] = mask_selected
            mask_selected = out

    return mask_selected.view(sizes)
