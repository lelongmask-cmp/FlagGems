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
# with a single scan over the whole tensor; beyond this size the multi-block
# pipeline with its count + scan + scatter + zero-fill launches wins.
SINGLE_PASS_THRESHOLD = 4096

# The scattered store runs at a fixed cost per lane on one vector core, so
# the only way to hide it is to spread lanes over all cores: target ~512
# blocks (a few waves) and clamp the block size to [128, 4096].
TARGET_NUM_BLOCKS = 512
MIN_BLOCK_SIZE = 128
MAX_BLOCK_SIZE = 4096

# Largest exclusive scan a single program handles. Above this many blocks the
# per-block counts are scanned hierarchically (chunk scan + totals scan +
# fixup), which only kicks in beyond ~32M elements at the maximum block size.
SCAN_NB_MAX = 8192


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
@triton.jit(do_not_specialize=["n_blocks"])
def masked_scatter_backward_scan_kernel(
    counts_ptr, n_blocks, NB: tl.constexpr
):
    # Single-program exclusive scan over the per-block counts: counts[0:nb]
    # becomes the exclusive prefix and counts[nb] the total. Store visibility
    # across launches works on the current driver, so no synchronize is
    # needed between the count kernel and this one.
    offsets = tl.arange(0, NB)
    m = offsets < n_blocks
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    pre = tl.cumsum(counts, axis=0)
    ex = pre - counts
    tl.store(counts_ptr + offsets, ex, mask=m)
    tl.store(counts_ptr + n_blocks, tl.sum(counts, axis=0))


@libentry()
@triton.jit(do_not_specialize=["n_blocks"])
def masked_scatter_backward_chunk_scan_kernel(
    counts_ptr, totals_ptr, n_blocks, NB: tl.constexpr
):
    # Hierarchical scan, level 1: exclusive scan within each chunk of the
    # counts array; the per-chunk totals go to totals_ptr.
    pid = tl.program_id(0)
    base = pid * NB
    offsets = base + tl.arange(0, NB)
    m = offsets < n_blocks
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    pre = tl.cumsum(counts, axis=0)
    ex = pre - counts
    tl.store(counts_ptr + offsets, ex, mask=m)
    tl.store(totals_ptr + pid, tl.sum(counts, axis=0))


@libentry()
@triton.jit(do_not_specialize=["n_blocks", "n_chunks"])
def masked_scatter_backward_chunk_fixup_kernel(
    counts_ptr, totals_ptr, n_blocks, n_chunks, NB: tl.constexpr
):
    # Hierarchical scan, level 3: add the exclusive chunk offsets (written by
    # the totals scan into totals_ptr) back into the per-block prefixes, and
    # copy the grand total to counts[n_blocks] where the scatter and
    # zero-fill kernels expect it. All programs store the same value; the
    # duplicate stores are idempotent.
    pid = tl.program_id(0)
    base = pid * NB
    offsets = base + tl.arange(0, NB)
    m = offsets < n_blocks
    advance = tl.load(totals_ptr + pid)
    prefix = tl.load(counts_ptr + offsets, mask=m, other=0) + advance
    tl.store(counts_ptr + offsets, prefix, mask=m)
    tl.store(counts_ptr + n_blocks, tl.load(totals_ptr + n_chunks))


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
    # compiler emit wrong stores below true_count (data-dependent corruption,
    # reproduced on the current driver).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < M
    true_count = tl.load(part_sums_ptr + num_blocks)
    zero_fill = block_mask & (offsets < N) & (offsets >= true_count)
    tl.store(out_ptr + offsets, 0, mask=zero_fill)


def masked_scatter_backward_impl(grad_output, mask, out, M, N):
    if M <= SINGLE_PASS_THRESHOLD:
        # Smallest launchable block is 8 lanes; anything below that is
        # handled by the bounds mask.
        block_size = max(8, triton.next_power_of_2(M))
        masked_scatter_backward_single_pass_kernel[(1,)](
            grad_output, mask, out, M, N, BLOCK_SIZE=block_size
        )
        return out

    BLOCK_SIZE = _pick_block_size(M)
    n_blocks = triton.cdiv(M, BLOCK_SIZE)

    with torch_device_fn.device(mask.device):
        dtype = torch.int32 if M < 2**31 else torch.int64
        counts = torch.empty(n_blocks + 1, dtype=dtype, device=mask.device)

        mask_count_per_block_kernel[(n_blocks,)](
            mask, counts, M, BLOCK_SIZE=BLOCK_SIZE
        )

        # Device-side exclusive prefix over the per-block counts, replacing
        # the previous host-side prefix / per-program prelude loops: store
        # visibility across launches works on the current driver, so the
        # whole pipeline runs without any host synchronization.
        n_chunks = triton.cdiv(n_blocks, SCAN_NB_MAX)
        if n_chunks == 1:
            masked_scatter_backward_scan_kernel[(1,)](
                counts, n_blocks, NB=triton.next_power_of_2(n_blocks)
            )
        else:
            totals = torch.empty(n_chunks + 1, dtype=dtype, device=mask.device)
            masked_scatter_backward_chunk_scan_kernel[(n_chunks,)](
                counts, totals, n_blocks, NB=SCAN_NB_MAX
            )
            masked_scatter_backward_scan_kernel[(1,)](
                totals, n_chunks, NB=triton.next_power_of_2(n_chunks)
            )
            masked_scatter_backward_chunk_fixup_kernel[(n_chunks,)](
                counts, totals, n_blocks, n_chunks, NB=SCAN_NB_MAX
            )

        masked_scatter_backward_kernel[(n_blocks,)](
            grad_output,
            mask,
            out,
            counts,
            M,
            n_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        masked_scatter_backward_zero_fill_kernel[(n_blocks,)](
            out,
            counts,
            M,
            N,
            n_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )

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
