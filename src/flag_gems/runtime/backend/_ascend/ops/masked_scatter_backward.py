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
# Ascend masked_scatter_backward — hybrid dispatch.
#
#   delegate  (N ≤ 67M):  delegate to Ascend masked_select (autotuned
#                          scatter-write kernel) + zero-pad.  The internal
#                          cumsum is grid-safe because ceil(N/1024) < 65536.
#                          This is 50-200× faster than the triton path for
#                          all sizes where cumsum works correctly.
#
#   triton    (N > 67M):  three-phase pure Triton.  The Ascend recursive
#                          cumsum's grid exceeds 65536 and produces corrupt
#                          results beyond this point.  The triton path is
#                          slower (scattered writes + per-CTA cumsum) but
#                          correct for arbitrary N.
#
# At 655M elements the triton write kernel's absolute latency (~3.2 s)
# is comparable to torch's aten implementation (~2.3 s), yielding a
# speedup near 1.0×.  At 67M the aten kernel is very fast (~0.24 s)
# while the triton write kernel is still ~0.2 s, so the gap is larger.
# ---------------------------------------------------------------------------

# ceil(67M / 1024) = 65536 = grid limit → cumsum breaks.
# Keep a margin: ceil(60M / 1024) ≈ 58594 < 65535.
_DELEGATE_MAX_N = 60 * 1024 * 1024  # 60M

_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 4096
_MAX_SCAN_BLOCK = 4096


# ---------------------------------------------------------------------------
# delegate path (N ≤ 60M)
# ---------------------------------------------------------------------------


def _delegate_path(grad_output, mask, numel):
    """Ascend masked_select + zero-pad (grid-safe for N ≤ 60M)."""
    from .masked_select import masked_select

    # int32 avoids cumsum bool→float32 precision loss
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
# triton path kernels (N > 60M)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _count_kernel(
    mask_ptr,
    counts_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_vals = tl.load(mask_ptr + offsets, mask=offsets < N, other=0)
    count = tl.sum(mask_vals.to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, count)


@libentry()
@triton.jit
def _scan_kernel(
    counts_ptr,
    part_sums_ptr,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks
    counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
    cumsums = tl.cumsum(counts, axis=0)
    tl.store(part_sums_ptr + offsets, cumsums - counts, mask=mask)


@libentry()
@triton.jit
def _chunk_scan_kernel(
    counts_ptr,
    part_sums_ptr,
    chunk_totals_ptr,
    n_blocks,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = offsets < n_blocks
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
    n_blocks,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = offsets < n_blocks
    val = tl.load(part_sums_ptr + offsets, mask=mask, other=0)
    chunk_offset = tl.load(chunk_offsets_ptr + pid)
    tl.store(part_sums_ptr + offsets, val + chunk_offset, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _write_kernel(
    grad_ptr,
    mask_ptr,
    part_sums_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
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


def _exclusive_scan(block_counts, n_blocks, device):
    part_sums = torch.empty(n_blocks, dtype=torch.int64, device=device)
    if n_blocks <= _MAX_SCAN_BLOCK:
        scan_block_size = triton.next_power_of_2(n_blocks)
        _scan_kernel[(1,)](
            block_counts, part_sums, n_blocks, BLOCK_SIZE=scan_block_size
        )
    else:
        n_chunks = triton.cdiv(n_blocks, _MAX_SCAN_BLOCK)
        chunk_totals = torch.empty(n_chunks, dtype=torch.int64, device=device)
        _chunk_scan_kernel[(n_chunks,)](
            block_counts, part_sums, chunk_totals, n_blocks,
            CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )
        chunk_offsets = torch.empty(n_chunks, dtype=torch.int64, device=device)
        scan_block2 = triton.next_power_of_2(n_chunks)
        _scan_kernel[(1,)](
            chunk_totals, chunk_offsets, n_chunks, BLOCK_SIZE=scan_block2,
        )
        _add_offsets_kernel[(n_chunks,)](
            part_sums, chunk_offsets, n_blocks, CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )
    return part_sums


def _triton_path(grad_output, mask, numel, N):
    """Three-phase Triton for N > 60M (cumsum grid-unsafe territory)."""
    BLOCK_SIZE = bracket_next_power_of_2(N, _MIN_BLOCK_SIZE, _MAX_BLOCK_SIZE)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    out = torch.zeros(numel, dtype=grad_output.dtype, device=grad_output.device)
    with torch_device_fn.device(grad_output.device):
        counts = torch.empty(n_blocks, dtype=torch.int64, device=mask.device)
        _count_kernel[(n_blocks,)](
            mask.ravel(), counts, N, BLOCK_SIZE=BLOCK_SIZE,
        )
        part_sums = _exclusive_scan(counts, n_blocks, mask.device)
        _write_kernel[(n_blocks,)](
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
    else:
        out = _triton_path(grad_output, mask, numel, N)

    return out.view(sizes)
