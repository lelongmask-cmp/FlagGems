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
# Ascend masked_scatter_backward — pure Triton.
#
#   fused  (N ≤ 4096):  single-CTA: zero-init + exclusive-cumsum + scatter.
#                        Uses torch.empty (no separate zeros kernel).
#
#   split  (N > 4096):  three-phase:
#                        1. per-block mask count (Triton)
#                        2. exclusive scan over counts (Triton, on-device)
#                        3. scatter-write into torch.zeros output
# ---------------------------------------------------------------------------

_MAX_FUSED_N = 4096
_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 4096
_MAX_SCAN_BLOCK = 4096


# ---------------------------------------------------------------------------
# fused — single-CTA for N ≤ 4096
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _fused_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    N,
    out_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused zero-init + exclusive-cumsum + scatter for small N."""
    offsets = tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N

    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    grad_val = tl.load(grad_ptr + offsets, mask=block_mask, other=0)

    pos = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1

    # Zero-init (single CTA → no cross-CTA race)
    zero_mask = offsets < out_numel
    tl.store(out_ptr + offsets, 0.0, mask=zero_mask)

    # Scatter gradient for True mask positions
    write_mask = block_mask & mask_val & (pos < out_numel)
    tl.store(out_ptr + pos, grad_val, mask=write_mask)


# ---------------------------------------------------------------------------
# split — multi-CTA for N > 4096
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _count_kernel(
    mask_ptr,
    counts_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Count True mask elements per block."""
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
    """Single-CTA exclusive scan."""
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
    """Per-chunk exclusive scan + store chunk total."""
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
    """Add per-chunk offsets to local exclusive scan results."""
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
    """Scatter-write into pre-zeroed output using per-block offsets."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N

    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    grad_val = tl.load(grad_ptr + offsets, mask=block_mask, other=0)

    global_offset = tl.load(part_sums_ptr + pid)
    local_pos = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    pos = global_offset + local_pos

    tl.store(out_ptr + pos, grad_val, mask=(block_mask & mask_val))


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _exclusive_scan(block_counts, n_blocks, device):
    """Exclusive scan over per-block counts, on-device."""
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


def _fused_path(grad_output, mask, numel, N):
    """Single-CTA for N ≤ 4096 (zero-init fused into kernel)."""
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty(numel, dtype=grad_output.dtype, device=grad_output.device)
    with torch_device_fn.device(grad_output.device):
        _fused_kernel[(1,)](grad_output, mask, out, N, numel, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _split_path(grad_output, mask, numel, N):
    """Multi-CTA for N > 4096.

    1. Count True mask elements per block
    2. Exclusive scan over block counts (on-device, single-CTA or two-level)
    3. Scatter-write into pre-zeroed output
    """
    BLOCK_SIZE = bracket_next_power_of_2(N, _MIN_BLOCK_SIZE, _MAX_BLOCK_SIZE)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)

    out = torch.zeros(numel, dtype=grad_output.dtype, device=grad_output.device)

    with torch_device_fn.device(grad_output.device):
        # 1. Count per block
        counts = torch.empty(n_blocks, dtype=torch.int64, device=mask.device)
        _count_kernel[(n_blocks,)](
            mask.ravel(), counts, N, BLOCK_SIZE=BLOCK_SIZE,
        )

        # 2. Exclusive scan (pure Triton, on-device)
        part_sums = _exclusive_scan(counts, n_blocks, mask.device)

        # 3. Scatter-write
        _write_kernel[(n_blocks,)](
            grad_output.ravel(), mask.ravel(), part_sums, out, N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def masked_scatter_backward(grad_output, mask, sizes):
    """
    Backward of masked_scatter w.r.t. ``source``.

    Matches aten::masked_scatter_backward(grad_output, mask, sizes) → Tensor.

    Two paths:
      - fused  (N ≤ 4096):  single-CTA Triton (zero-init + cumsum + scatter)
      - split  (N > 4096):  count + exclusive scan + scatter-write
    """
    logger.debug("GEMS_ASCEND MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()

    if N <= _MAX_FUSED_N:
        out = _fused_path(grad_output, mask, numel, N)
    else:
        out = _split_path(grad_output, mask, numel, N)

    return out.view(sizes)
