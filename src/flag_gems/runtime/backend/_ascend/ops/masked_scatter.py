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
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def masked_scatter_single_pass_kernel(
    inp_ptr, mask_ptr, src_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    block_mask = offsets < N

    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)

    mask_ints = mask_val.to(tl.int32)
    src_indices = tl.cumsum(mask_ints, axis=0) - 1

    active = block_mask & mask_val
    src_val = tl.load(src_ptr + src_indices, mask=active)
    tl.store(inp_ptr + offsets, src_val, mask=active)


@libentry()
@triton.jit
def count_mask_per_block_kernel(mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offset < N
    mask_val = tl.load(mask_ptr + offset, mask=block_mask, other=0).to(tl.int32)
    count = tl.sum(mask_val)
    tl.store(counts_ptr + pid, count)


@libentry()
@triton.jit(do_not_specialize=["N", "num_blocks", "num_blocks_per_row"])
def masked_scatter_kernel(
    inp_ptr,
    mask_ptr,
    src_ptr,
    part_sums_ptr,
    N,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)

    start_block = row_id * num_blocks_per_row
    offset = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    advance = tl.load(part_sums_ptr + row_id)

    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)

    for block_id in range(start_block, last_block_id):
        select_mask = tl.load(mask_ptr + offset).to(tl.int1)
        select_ints = select_mask.to(tl.int32)

        block_cumsum = tl.cumsum(select_ints, axis=0) - 1
        global_src_idx = advance + block_cumsum

        advance += tl.sum(select_ints, axis=0)

        src_val = tl.load(src_ptr + global_src_idx, mask=select_mask)
        tl.store(inp_ptr + offset, src_val, mask=select_mask)

        offset += BLOCK_SIZE

    block_mask = offset < N
    select_mask = tl.load(mask_ptr + offset, mask=block_mask, other=0).to(tl.int1)

    select_ints = select_mask.to(tl.int32)
    block_cumsum = tl.cumsum(select_ints, axis=0) - 1
    global_src_idx = advance + block_cumsum

    active = block_mask & select_mask
    src_val = tl.load(src_ptr + global_src_idx, mask=active)
    tl.store(inp_ptr + offset, src_val, mask=active)


@libentry()
@triton.jit
def masked_scatter_backward_single_pass_kernel(
    grad_ptr, mask_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < M

    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)

    mask_ints = mask_val.to(tl.int32)
    out_indices = tl.cumsum(mask_ints, axis=0) - 1

    # Gather grad values at the mask positions (contiguous reads) and compact
    # them into the dense out positions prefix_sum - 1, so the active lanes of
    # every store hit consecutive addresses instead of scattered ones.
    src_val = tl.load(grad_ptr + offsets, mask=mask_val, other=0)
    tl.store(
        out_ptr + out_indices, src_val, mask=mask_val & (out_indices < N)
    )

    # The composite also zero-fills out positions [true_count, N) regardless
    # of the mask; fold that into the same pass to avoid the separate zeros +
    # cat kernels the native implementation pays for. The two stores address
    # disjoint ranges [0, true_count) and [true_count, N).
    true_count = tl.sum(mask_ints, axis=0)
    zero_fill = block_mask & (offsets < N) & (offsets >= true_count)
    tl.store(out_ptr + offsets, 0, mask=zero_fill)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "num_blocks"])
def masked_scatter_backward_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    part_sums_ptr,
    M,
    N,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < M

    select_mask = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)

    select_ints = select_mask.to(tl.int32)
    block_cumsum = tl.cumsum(select_ints, axis=0) - 1
    out_indices = tl.load(part_sums_ptr + pid) + block_cumsum

    src_val = tl.load(grad_ptr + offsets, mask=select_mask, other=0)
    tl.store(
        out_ptr + out_indices, src_val, mask=select_mask & (out_indices < N)
    )

    true_count = tl.load(part_sums_ptr + num_blocks)
    zero_fill = block_mask & (offsets < N) & (offsets >= true_count)
    tl.store(out_ptr + offsets, 0, mask=zero_fill)


def masked_scatter_backward_impl(grad_output, mask, out, M, N):
    if M <= 4096:
        # Keep at least 32 lanes: masked stores with the zero-fill second
        # store miscompile on this backend for very small blocks.
        BLOCK_SIZE = max(32, triton.next_power_of_2(M))
        masked_scatter_backward_single_pass_kernel[(1,)](
            grad_output, mask, out, M, N, BLOCK_SIZE=BLOCK_SIZE
        )
        return out

    BLOCK_SIZE = bracket_next_power_of_2(M, 128, 4096)
    n_blocks = triton.cdiv(M, BLOCK_SIZE)

    with torch_device_fn.device(mask.device):
        block_counts = torch.empty(n_blocks, dtype=torch.int64, device=mask.device)
        count_mask_per_block_kernel[(n_blocks,)](
            mask, block_counts, M, BLOCK_SIZE=BLOCK_SIZE
        )

        counts_cpu = block_counts.cpu().to(torch.int64)
        # part_sums[b] is the number of selected elements before block b and
        # part_sums[n_blocks] is the total number of selected elements.
        part_sums = torch.zeros(n_blocks + 1, dtype=torch.int64)
        torch.cumsum(counts_cpu, dim=0, out=part_sums[1:])
        part_sums = part_sums.to(mask.device)

        masked_scatter_backward_kernel[(n_blocks,)](
            grad_output,
            mask,
            out,
            part_sums,
            M,
            N,
            n_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


def masked_scatter_impl(inp, mask, source, N):
    true_count = mask.sum().item()
    if true_count == 0:
        return inp

    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(N)
        masked_scatter_single_pass_kernel[(1,)](
            inp, mask, source, N, BLOCK_SIZE=BLOCK_SIZE
        )
        return inp

    BLOCK_SIZE = bracket_next_power_of_2(N, 128, 4096)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)

    with torch_device_fn.device(inp.device):
        block_counts = torch.empty(n_blocks, dtype=torch.int64, device=mask.device)
        count_mask_per_block_kernel[(n_blocks,)](
            mask, block_counts, N, BLOCK_SIZE=BLOCK_SIZE
        )

        counts_cpu = block_counts.cpu().to(torch.int64)
        prefix_sum = torch.zeros(n_blocks, dtype=torch.int64)
        torch.cumsum(counts_cpu[:-1], dim=0, out=prefix_sum[1:])
        part_sums = prefix_sum.to(mask.device)

        masked_scatter_kernel[(n_blocks,)](
            inp,
            mask,
            source,
            part_sums,
            N,
            n_blocks,
            1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return inp


def masked_scatter(inp, mask, source):
    logger.debug("GEMS_ASCEND MASKED_SCATTER")

    assert broadcastable(
        inp.shape, mask.shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"

    _, mask = torch.broadcast_tensors(inp, mask)

    out = inp.clone()
    if not out.is_contiguous():
        out = out.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    if not source.is_contiguous():
        source = source.contiguous()

    N = out.numel()

    masked_scatter_impl(out, mask, source, N)

    return out


def masked_scatter_(inp, mask, source):
    logger.debug("GEMS_ASCEND MASKED_SCATTER_")

    assert broadcastable(inp.shape, mask.shape)
    _, mask = torch.broadcast_tensors(inp, mask)

    if not inp.is_contiguous():
        raise RuntimeError(
            "in-place operation currently requires contiguous input tensor. "
        )

    mask = mask if mask.is_contiguous() else mask.contiguous()
    source = source if source.is_contiguous() else source.contiguous()

    N = inp.numel()
    masked_scatter_impl(inp, mask, source, N)

    return inp


def masked_scatter_backward(grad_output, mask, sizes):
    logger.debug("GEMS_ASCEND MASKED_SCATTER_BACKWARD")

    assert broadcastable(
        grad_output.shape, mask.shape
    ), "The shapes of the `mask` and the `grad_output` tensor must be broadcastable"

    # Mirror the native composite: the mask is expanded to grad_output's shape
    # only when it is smaller; the autograd engine passes an already expanded
    # mask, so this is a no-op on the real backward path.
    if mask.numel() < grad_output.numel():
        mask = mask.expand(grad_output.shape).contiguous()
    else:
        mask = mask.contiguous()
    grad_output = grad_output.contiguous()

    N = 1
    for s in sizes:
        N *= int(s)

    if N == 0:
        return torch.empty(
            list(sizes), dtype=grad_output.dtype, device=grad_output.device
        )

    M = mask.numel()

    if N > M:
        # Positions beyond the mask are never written by the kernel; the native
        # composite zero-fills them through zeros + cat, pre-fill them instead.
        out = torch.zeros(
            list(sizes), dtype=grad_output.dtype, device=grad_output.device
        )
    else:
        out = torch.empty(
            list(sizes), dtype=grad_output.dtype, device=grad_output.device
        )

    masked_scatter_backward_impl(grad_output, mask, out, M, N)

    return out
