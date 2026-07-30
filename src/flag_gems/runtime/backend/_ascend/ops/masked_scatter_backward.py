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
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ascend masked_scatter_backward
#
#   single-tile (N ≤ TILE): one fused kernel: bool mask read → cumsum →
#                            zero-init → scatter.  Zero overhead.
#
#   multi-tile  (TILE < N ≤ 80M): tile_cumsum (bool input) + scan +
#                                  tile_update + autotuned scatter + zero-pad.
#
#   triton      (N > 80M): count + scan + scatter-write (memory-efficient).
# ---------------------------------------------------------------------------

_TILE_SIZE = 4096
_TWOLEVEL_MAX_N = 80 * 1024 * 1024
_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 4096
_MAX_SCAN_BLOCK = 4096


# ---------------------------------------------------------------------------
# single-tile fused kernel (N ≤ TILE_SIZE)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _fused_kernel(
    grad_ptr, mask_ptr, out_ptr, N, out_numel, BLOCK_SIZE: tl.constexpr,
):
    """Single-CTA: read bool mask → cumsum → zero-init → scatter.

    No mask copy, no separate zeros, no slice copy — everything in one launch.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N

    # Read bool mask, convert to int32 for cumsum, int1 for mask ops
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0)
    mask_int = mask_val.to(tl.int32)
    pos = tl.cumsum(mask_int, axis=0) - 1

    grad_val = tl.load(grad_ptr + offsets, mask=block_mask, other=0)

    # Zero-init output (single CTA → no cross-CTA race)
    tl.store(out_ptr + offsets, 0.0, mask=(offsets < out_numel))

    # Scatter: only at True mask positions where block is valid and pos in range
    tl.store(out_ptr + pos, grad_val,
             mask=block_mask & mask_val.to(tl.int1) & (pos >= 0) & (pos < out_numel))


# ---------------------------------------------------------------------------
# multi-tile kernels (TILE < N ≤ 80M)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tile_cumsum_kernel(
    inp_ptr, out_ptr, tile_totals_ptr, N, TILE_SIZE: tl.constexpr,
):
    """Per-tile cumsum.  Accepts bool input, converts to int32 internally."""
    pid = ext.program_id(axis=0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    m = offsets < N
    x = tl.load(inp_ptr + offsets, mask=m, other=0).to(tl.int32)
    cumsum = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offsets, cumsum, mask=m)
    tl.store(tile_totals_ptr + pid, tl.sum(x, axis=0))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tile_update_kernel(
    out_ptr, tile_offsets_ptr, N, TILE_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    m = offsets < N
    val = tl.load(out_ptr + offsets, mask=m, other=0)
    tl.store(out_ptr + offsets, val + tl.load(tile_offsets_ptr + pid), mask=m)


@libentry()
@triton.jit
def _scan_kernel(counts_ptr, part_sums_ptr, n_elem, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    m = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=m)


@libentry()
@triton.jit
def _chunk_scan_kernel(
    counts_ptr, part_sums_ptr, chunk_totals_ptr, n_elem, CHUNK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=m)
    tl.store(chunk_totals_ptr + pid, tl.sum(counts, axis=0))


@libentry()
@triton.jit
def _add_offsets_kernel(
    part_sums_ptr, chunk_offsets_ptr, n_elem, CHUNK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem
    val = tl.load(part_sums_ptr + offsets, mask=m, other=0)
    tl.store(part_sums_ptr + offsets, val + tl.load(chunk_offsets_ptr + pid), mask=m)


def _exclusive_scan(arr, n_elems, device):
    part_sums = torch.empty(n_elems, dtype=torch.int64, device=device)
    if n_elems <= _MAX_SCAN_BLOCK:
        scan_block = triton.next_power_of_2(n_elems)
        _scan_kernel[(1,)](arr, part_sums, n_elems, BLOCK_SIZE=scan_block)
    else:
        n_chunks = triton.cdiv(n_elems, _MAX_SCAN_BLOCK)
        chunk_totals = torch.empty(n_chunks, dtype=torch.int64, device=device)
        _chunk_scan_kernel[(n_chunks,)](
            arr, part_sums, chunk_totals, n_elems, CHUNK_SIZE=_MAX_SCAN_BLOCK,
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


def _multi_tile_path(grad_output, mask, numel, N):
    """Tile cumsum (bool input) + scan + tile_update + autotuned scatter."""
    device = grad_output.device
    with torch_device_fn.device(device):
        mask_flat = mask.ravel()
        n_tiles = triton.cdiv(N, _TILE_SIZE)

        # 1. tile cumsum (reads bool directly, no int32 copy)
        prefix_sum = torch.empty(N, dtype=torch.int32, device=device)
        tile_totals = torch.empty(n_tiles, dtype=torch.int64, device=device)
        _tile_cumsum_kernel[(n_tiles,)](
            mask_flat, prefix_sum, tile_totals, N, TILE_SIZE=_TILE_SIZE,
        )

        # 2. scan + tile_update → full 1-based prefix sum
        tile_offsets = _exclusive_scan(tile_totals, n_tiles, device)
        _tile_update_kernel[(n_tiles,)](
            prefix_sum, tile_offsets, N, TILE_SIZE=_TILE_SIZE,
        )

        # 3. autotuned scatter (uses full prefix sum, no per-CTA tile lookup)
        n_selected = prefix_sum[-1].item()
        mask_selected = torch.empty(n_selected, dtype=grad_output.dtype, device=device)
        from .masked_select import masked_select_kernel
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        masked_select_kernel[grid](
            grad_output.ravel(), mask_flat.to(torch.int32), prefix_sum, mask_selected, N,
        )

    if n_selected < numel:
        out = torch.zeros(numel, dtype=mask_selected.dtype, device=device)
        out[:n_selected] = mask_selected
        return out
    return mask_selected


# ---------------------------------------------------------------------------
# triton path (N > 80M)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tr_count_kernel(mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(counts_ptr + pid, tl.sum(
        tl.load(mask_ptr + offsets, mask=offsets < N, other=0).to(tl.int32), 0))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tr_write_kernel(
    grad_ptr, mask_ptr, part_sums_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    grad_val = tl.load(grad_ptr + offsets, mask=block_mask, other=0)
    global_offset = tl.load(part_sums_ptr + pid)
    local_pos = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    tl.store(out_ptr + global_offset + local_pos, grad_val, mask=(block_mask & mask_val))


def _triton_fallback(grad_output, mask, numel, N):
    BLOCK_SIZE = bracket_next_power_of_2(N, _MIN_BLOCK_SIZE, _MAX_BLOCK_SIZE)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    device = grad_output.device
    out = torch.zeros(numel, dtype=grad_output.dtype, device=device)
    with torch_device_fn.device(device):
        counts = torch.empty(n_blocks, dtype=torch.int64, device=device)
        _tr_count_kernel[(n_blocks,)](mask.ravel(), counts, N, BLOCK_SIZE=BLOCK_SIZE)
        part_sums = _exclusive_scan(counts, n_blocks, device)
        _tr_write_kernel[(n_blocks,)](grad_output.ravel(), mask.ravel(), part_sums, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------


def masked_scatter_backward(grad_output, mask, sizes):
    logger.debug("GEMS_ASCEND MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()

    if N <= _TILE_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(N)
        out = torch.empty(numel, dtype=grad_output.dtype, device=grad_output.device)
        with torch_device_fn.device(grad_output.device):
            _fused_kernel[(1,)](grad_output.ravel(), mask.ravel(), out, N, numel, BLOCK_SIZE=BLOCK_SIZE)
    elif N <= _TWOLEVEL_MAX_N:
        out = _multi_tile_path(grad_output, mask, numel, N)
    else:
        out = _triton_fallback(grad_output, mask, numel, N)

    return out.view(sizes)
