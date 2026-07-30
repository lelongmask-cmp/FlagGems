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
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Three-way hybrid dispatch based on N = mask.numel():
#
#   delegate (N ≤ 2M):     Ascend masked_select (autotuned, grid-safe)
#                           + zero-pad.  2-3 launches, fastest for small N.
#
#   twolevel (2M < N ≤ 80M): normed_cumsum-style two-level prefix sum
#                             + autotuned scatter-write + zero-pad.
#                             5-7 launches, excellent for medium-large N
#                             where the prefix sum buffer is manageable.
#
#   triton   (N > 80M):     three-phase count + exclusive scan + scatter-write.
#                           Memory-efficient (only per-block counts, not
#                           full prefix sum).  Correct for arbitrary N.
# ---------------------------------------------------------------------------

_DELEGATE_MAX_N = 2 * 1024 * 1024
_TWOLEVEL_MAX_N = 80 * 1024 * 1024

_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 4096
_MAX_SCAN_BLOCK = 4096
_TILE_SIZE = 4096


# ---------------------------------------------------------------------------
# delegate path (N ≤ 2M)
# ---------------------------------------------------------------------------


def _delegate_path(grad_output, mask, numel):
    from .masked_select import masked_select

    mask_selected = masked_select(grad_output, mask.to(torch.int32))
    n_selected = mask_selected.numel()
    if n_selected < numel:
        out = torch.zeros(
            numel, dtype=mask_selected.dtype, device=mask_selected.device
        )
        out[:n_selected] = mask_selected
        return out
    return mask_selected


# ---------------------------------------------------------------------------
# twolevel path kernels (2M < N ≤ 80M)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tl_tile_cumsum_kernel(
    inp_ptr, out_ptr, tile_totals_ptr, N, TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offsets < N
    x = tl.load(inp_ptr + offsets, mask=mask, other=0)
    cumsum = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offsets, cumsum, mask=mask)
    total = tl.sum(x, axis=0)
    tl.store(tile_totals_ptr + pid, total)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tl_tile_update_kernel(
    out_ptr, tile_offsets_ptr, N, TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offsets < N
    val = tl.load(out_ptr + offsets, mask=mask, other=0)
    offset = tl.load(tile_offsets_ptr + pid)
    tl.store(out_ptr + offsets, val + offset, mask=mask)


@libentry()
@triton.jit
def _tl_scan_kernel(counts_ptr, part_sums_ptr, n_elem, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    m = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=m)


@libentry()
@triton.jit
def _tl_chunk_scan_kernel(
    counts_ptr, part_sums_ptr, chunk_totals_ptr, n_elem, CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=m)
    total = tl.sum(counts, axis=0)
    tl.store(chunk_totals_ptr + pid, total)


@libentry()
@triton.jit
def _tl_add_offsets_kernel(
    part_sums_ptr, chunk_offsets_ptr, n_elem, CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem
    val = tl.load(part_sums_ptr + offsets, mask=m, other=0)
    chunk_offset = tl.load(chunk_offsets_ptr + pid)
    tl.store(part_sums_ptr + offsets, val + chunk_offset, mask=m)


def _tl_exclusive_scan(arr, n_elems, device):
    part_sums = torch.empty(n_elems, dtype=torch.int64, device=device)
    if n_elems <= _MAX_SCAN_BLOCK:
        scan_block = triton.next_power_of_2(n_elems)
        _tl_scan_kernel[(1,)](arr, part_sums, n_elems, BLOCK_SIZE=scan_block)
    else:
        n_chunks = triton.cdiv(n_elems, _MAX_SCAN_BLOCK)
        chunk_totals = torch.empty(n_chunks, dtype=torch.int64, device=device)
        _tl_chunk_scan_kernel[(n_chunks,)](
            arr, part_sums, chunk_totals, n_elems, CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )
        chunk_offsets = torch.empty(n_chunks, dtype=torch.int64, device=device)
        scan_block2 = triton.next_power_of_2(n_chunks)
        _tl_scan_kernel[(1,)](
            chunk_totals, chunk_offsets, n_chunks, BLOCK_SIZE=scan_block2,
        )
        _tl_add_offsets_kernel[(n_chunks,)](
            part_sums, chunk_offsets, n_elems, CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )
    return part_sums


def _twolevel_path(grad_output, mask, numel, N):
    """Two-level prefix sum + autotuned scatter-write."""
    device = grad_output.device
    with torch_device_fn.device(device):
        mask_flat = mask.ravel().to(torch.int32)
        n_tiles = triton.cdiv(N, _TILE_SIZE)

        # 1. tile cumsum
        prefix_sum = torch.empty(N, dtype=torch.int32, device=device)
        tile_totals = torch.empty(n_tiles, dtype=torch.int64, device=device)
        _tl_tile_cumsum_kernel[(n_tiles,)](
            mask_flat, prefix_sum, tile_totals, N, TILE_SIZE=_TILE_SIZE,
        )

        # 2. scan tile totals → tile offsets
        tile_offsets = _tl_exclusive_scan(tile_totals, n_tiles, device)

        # 3. add tile offsets
        _tl_tile_update_kernel[(n_tiles,)](
            prefix_sum, tile_offsets, N, TILE_SIZE=_TILE_SIZE,
        )

        # 4. autotuned scatter-write
        n_selected = prefix_sum[-1].item()
        mask_selected = torch.empty(
            n_selected, dtype=grad_output.dtype, device=device
        )
        from .masked_select import masked_select_kernel
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        masked_select_kernel[grid](
            grad_output.ravel(), mask_flat, prefix_sum, mask_selected, N,
        )

    # 5. zero-pad
    if n_selected < numel:
        out = torch.zeros(numel, dtype=mask_selected.dtype, device=device)
        out[:n_selected] = mask_selected
        return out
    return mask_selected


# ---------------------------------------------------------------------------
# triton path kernels (N > 80M, memory-efficient)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tr_count_kernel(
    mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_vals = tl.load(mask_ptr + offsets, mask=offsets < N, other=0)
    count = tl.sum(mask_vals.to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, count)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tr_write_kernel(
    grad_ptr, mask_ptr, part_sums_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    grad_val = tl.load(grad_ptr + offsets, mask=block_mask, other=0)
    global_offset = tl.load(part_sums_ptr + pid)
    local_pos = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    pos = global_offset + local_pos
    tl.store(out_ptr + pos, grad_val, mask=(block_mask & mask_val))


def _triton_path(grad_output, mask, numel, N):
    """Count + exclusive scan + scatter-write (memory-efficient)."""
    BLOCK_SIZE = bracket_next_power_of_2(N, _MIN_BLOCK_SIZE, _MAX_BLOCK_SIZE)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    device = grad_output.device
    out = torch.zeros(numel, dtype=grad_output.dtype, device=device)
    with torch_device_fn.device(device):
        counts = torch.empty(n_blocks, dtype=torch.int64, device=device)
        _tr_count_kernel[(n_blocks,)](
            mask.ravel(), counts, N, BLOCK_SIZE=BLOCK_SIZE,
        )
        part_sums = _tl_exclusive_scan(counts, n_blocks, device)
        _tr_write_kernel[(n_blocks,)](
            grad_output.ravel(), mask.ravel(), part_sums, out, N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def masked_scatter_backward(grad_output, mask, sizes):
    """Backward of masked_scatter w.r.t. ``source``."""
    logger.debug("GEMS_ASCEND MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()

    if N <= _DELEGATE_MAX_N:
        out = _delegate_path(grad_output, mask, numel)
    elif N <= _TWOLEVEL_MAX_N:
        out = _twolevel_path(grad_output, mask, numel, N)
    else:
        out = _triton_path(grad_output, mask, numel, N)

    return out.view(sizes)
