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

logger = logging.getLogger(__name__)

# Use the single-pass kernel up to this many elements. It runs one program
# with a single scan over the whole tensor; the count + prefix + scatter
# pipeline pays two extra launches and two synchronizations, which only
# amortize beyond this size.
SINGLE_PASS_THRESHOLD = 4096

# The scattered store runs at a fixed cost per lane on one vector core, so
# the only way to hide it is to spread lanes over all cores: target ~512
# blocks (a few waves of 40 cores) and clamp the block size to [128, 4096].
TARGET_NUM_BLOCKS = 512
MIN_BLOCK_SIZE = 128
MAX_BLOCK_SIZE = 4096


def _pick_block_size(M):
    block_size = triton.next_power_of_2(triton.cdiv(M, TARGET_NUM_BLOCKS))
    return min(max(block_size, MIN_BLOCK_SIZE), MAX_BLOCK_SIZE)


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
@triton.jit(do_not_specialize=["M"])
def mask_count_per_block_kernel(mask_ptr, counts_ptr, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < M
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int32)
    tl.store(counts_ptr + pid, tl.sum(mask_val, axis=0))


@libentry()
@triton.jit(do_not_specialize=["M", "num_blocks"])
def masked_scatter_backward_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    part_sums_ptr,
    M,
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
    # out_indices is always < true_count <= N (the tail is zero-filled by a
    # dedicated kernel), hence the plain selection mask.
    tl.store(out_ptr + out_indices, src_val, mask=select_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "num_blocks"])
def masked_scatter_backward_zero_fill_kernel(
    out_ptr, part_sums_ptr, M, N, num_blocks, BLOCK_SIZE: tl.constexpr
):
    # Zeroes out[true_count:N). Kept in its own kernel: combining this dense
    # masked store with the scattered store of the main kernel makes the
    # compiler emit stores below true_count on this backend.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < M
    true_count = tl.load(part_sums_ptr + num_blocks)
    zero_fill = block_mask & (offsets < N) & (offsets >= true_count)
    tl.store(out_ptr + offsets, 0, mask=zero_fill)


@libentry()
@triton.jit(do_not_specialize=["M", "num_blocks"])
def masked_scatter_backward_kernel_with_prelude(
    grad_ptr,
    mask_ptr,
    out_ptr,
    counts_ptr,
    M,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    # Variant for few blocks: the exclusive prefix is recomputed per program
    # with a short scalar loop over the per-block counts, so the pipeline
    # needs no device-side prefix step at all (torch.cumsum/cat dispatch to
    # the flag-gems kernels under use_gems and are an order of magnitude
    # slower than this loop for a few hundred blocks).
    pid = tl.program_id(0)
    advance = 0
    for i in range(0, num_blocks):
        count = tl.load(counts_ptr + i)
        advance += tl.where(i < pid, count, 0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < M

    select_mask = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)

    select_ints = select_mask.to(tl.int32)
    block_cumsum = tl.cumsum(select_ints, axis=0) - 1
    out_indices = advance + block_cumsum

    src_val = tl.load(grad_ptr + offsets, mask=select_mask, other=0)
    tl.store(out_ptr + out_indices, src_val, mask=select_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "num_blocks"])
def masked_scatter_backward_zero_fill_kernel_with_prelude(
    out_ptr, counts_ptr, M, N, num_blocks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    true_count = 0
    for i in range(0, num_blocks):
        true_count += tl.load(counts_ptr + i)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < M
    zero_fill = block_mask & (offsets < N) & (offsets >= true_count)
    tl.store(out_ptr + offsets, 0, mask=zero_fill)


# Above this many blocks the per-program prefix loop in the *_with_prelude
# kernels costs more than a host-side prefix over the counts.
MAX_BLOCKS_FOR_PRELUDE = 512


def masked_scatter_backward_impl(grad_output, mask, out, M, N):
    if M <= SINGLE_PASS_THRESHOLD:
        # Keep at least 32 lanes: masked stores with the zero-fill second
        # store miscompile on this backend for very small blocks.
        block_size = max(32, triton.next_power_of_2(M))
        masked_scatter_backward_single_pass_kernel[(1,)](
            grad_output, mask, out, M, N, BLOCK_SIZE=block_size
        )
        return out

    BLOCK_SIZE = _pick_block_size(M)
    n_blocks = triton.cdiv(M, BLOCK_SIZE)

    with torch_device_fn.device(mask.device):
        dtype = torch.int32 if M < 2**31 else torch.int64
        counts = torch.empty(n_blocks, dtype=dtype, device=mask.device)

        mask_count_per_block_kernel[(n_blocks,)](
            mask, counts, M, BLOCK_SIZE=BLOCK_SIZE
        )

        # Triton kernel stores on this backend are not guaranteed to be
        # visible to work enqueued after the launch, so synchronize before
        # anything else consumes `counts`.
        torch_device_fn.synchronize()

        if n_blocks <= MAX_BLOCKS_FOR_PRELUDE:
            masked_scatter_backward_kernel_with_prelude[(n_blocks,)](
                grad_output,
                mask,
                out,
                counts,
                M,
                n_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            masked_scatter_backward_zero_fill_kernel_with_prelude[(n_blocks,)](
                out,
                counts,
                M,
                N,
                n_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # part_sums[b] is the number of selected elements before block b
            # and part_sums[n_blocks] is the total. A host-side prefix over a
            # few thousand ints is far cheaper than the intercepted
            # torch.cumsum/torch.cat kernels under use_gems.
            counts_cpu = counts.cpu().to(torch.int64).tolist()
            prefix = [0]
            running = 0
            for c in counts_cpu[:-1]:
                running += c
                prefix.append(running)
            prefix.append(running + counts_cpu[-1])
            part_sums = torch.tensor(
                prefix, dtype=dtype, device=mask.device
            )

            masked_scatter_backward_kernel[(n_blocks,)](
                grad_output,
                mask,
                out,
                part_sums,
                M,
                n_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            masked_scatter_backward_zero_fill_kernel[(n_blocks,)](
                out,
                part_sums,
                M,
                N,
                n_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        # Same visibility caveat: make the stores visible before any
        # downstream consumer reads `out`.
        torch_device_fn.synchronize()

    return out


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

    out = torch.empty(
        list(sizes), dtype=grad_output.dtype, device=grad_output.device
    )

    masked_scatter_backward_impl(grad_output, mask, out, M, N)

    return out
