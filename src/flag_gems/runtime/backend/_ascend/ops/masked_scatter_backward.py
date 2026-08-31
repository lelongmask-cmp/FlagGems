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
import torch_npu  # noqa: F401  (native kernels for the .out dispatch below)
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
#   Semantics: out = cat([grad[mask], zeros]).view(sizes)
#     i.e.  out[j] = grad[idx[j]] for j < k, out[j] = 0 for k <= j < numel,
#     where idx = sorted True positions of mask and k = #True.
#
#   small N   (#blocks <= 128): ONE kernel does everything — each program
#                            re-reads earlier mask blocks (dense vector
#                            loads, torch-written -> no sync needed) for its
#                            exclusive offset, scatters its block, and the
#                            last program zero-fills the disjoint tail
#                            [total, numel).
#
#   large N   (#blocks > 128): Triton count kernel (~150µs) -> k on host,
#                            torch_npu's native nonzero via the .out
#                            overload (aten::nonzero.out — gems only
#                            overrides aten::nonzero, so this stays on the
#                            fast C++/CANN path, ~1.7ms @16M) fills idx with
#                            the sorted True positions, then ONE Triton
#                            kernel iterates the COMPACT output range:
#                              dense load idx -> gather load grad[idx]
#                              -> affine store out[j], plus affine zero-fill
#                              of [k, numel).  All masks are affine — a
#                              data-dependent store mask (e.g. idx[j] < N
#                              sentinel check) compiles to per-lane slow
#                              stores (~8x worse); data-dependent load
#                              ADDRESSES are fine.
#                            A scattered GATHER load measures ~40ns/lane on
#                            910B — ~5x cheaper than the scattered STORE
#                            (~200ns/lane) the old count->scan->scatter
#                            formulation used (24ms @16M; the flip is ~5ms).
#
# Measured platform costs (910B4, triton-ascend 3.2.0):
#   kernel launch ~57µs device time, stream sync ~7µs,
#   scattered store ~200ns/lane, scattered gather load ~40ns/lane,
#   dense stores are HBM-bandwidth fast.
# ---------------------------------------------------------------------------

_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 1024
_MAX_REREAD_BLOCKS = 128
_REREAD_CHUNK = 4096  # (must be restated literally inside jit kernels)


@libentry()
@triton.jit(do_not_specialize=["N", "numel", "NB"])
def _scatter_reread_kernel(
    grad_ptr, mask_ptr, out_ptr, N, numel, NB, BLOCK_SIZE: tl.constexpr,
):
    """Single-launch multi-CTA scatter + zero-fill.

    Program p computes its exclusive offset = sum(mask[0 : p*BLOCK)) with
    dense vector chunk loads (reads only torch-written data, so no scan
    kernel and no sync are needed).  The last program additionally zero-fills
    the tail [total, numel) — that region is disjoint from every scatter
    write, so there is no cross-program race.  Used when #blocks is small
    enough that the re-read traffic (O(N * #blocks)) stays cheap.
    """
    pid = ext.program_id(axis=0)
    CHUNK: tl.constexpr = 4096  # dense chunk for offset re-read
    total_prev = pid * BLOCK_SIZE
    off = tl.cast(0, tl.int64)
    n_chunks = total_prev // CHUNK
    for i in range(n_chunks):
        offs = i * CHUNK + tl.arange(0, CHUNK)
        off += tl.sum(tl.load(mask_ptr + offs, mask=offs < N, other=0).to(tl.int32),
                      axis=0)
    rem_start = n_chunks * CHUNK
    rem = total_prev - rem_start
    if rem > 0:
        offs = rem_start + tl.arange(0, CHUNK)
        off += tl.sum(
            tl.load(mask_ptr + offs, mask=(offs < N) & (offs < total_prev), other=0)
            .to(tl.int32), axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    grad_val = tl.load(grad_ptr + offsets, mask=m, other=0)
    pos = off + tl.cumsum(mask_val, axis=0) - 1
    tl.store(out_ptr + pos, grad_val, mask=m & (mask_val == 1) & (pos < numel))

    # Last program: zero-fill tail [total, numel) — disjoint from all scatters.
    if pid == NB - 1:
        total = off + tl.sum(mask_val, axis=0)
        n_zc = (numel - total) // CHUNK
        for i in range(n_zc):
            zoffs = total + i * CHUNK + tl.arange(0, CHUNK)
            tl.store(out_ptr + zoffs, 0.0, mask=zoffs < numel)
        zrem = total + n_zc * CHUNK
        if zrem < numel:
            zoffs = zrem + tl.arange(0, CHUNK)
            tl.store(out_ptr + zoffs, 0.0, mask=(zoffs < numel) & (zoffs >= total))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _count_kernel(mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Per-block True counts of the mask (dense read, ~150µs @16M)."""
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    tl.store(counts_ptr + pid, tl.sum(mask_val, axis=0))


@libentry()
@triton.jit(do_not_specialize=["k", "numel"])
def _flip_kernel(
    grad_ptr, idx_ptr, out_ptr, k, numel, BLOCK_SIZE: tl.constexpr,
):
    """One kernel over the compact output range.

    idx holds the sorted True positions of the mask (k entries).
      out[j] = grad[idx[j]]  for j < k      (dense load of idx, scattered
                                            GATHER of grad, affine store)
      out[j] = 0             for k <= j < numel
    Every mask here is affine in the lane index — data-dependent store
    masks compile to per-lane slow stores on this backend, so the lane
    validity must come from the scalar k, never from loaded data.  The two
    stores have disjoint masks, so every output element is written exactly
    once by one lane — no sync, no separate zero-fill launch.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < k
    pos = tl.load(idx_ptr + offsets, mask=m, other=0)
    val = tl.load(grad_ptr + pos, mask=m, other=0)
    tl.store(out_ptr + offsets, val, mask=m)
    tl.store(out_ptr + offsets, 0.0, mask=(offsets >= k) & (offsets < numel))


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

    BLOCK_SIZE = bracket_next_power_of_2(
        triton.cdiv(N, 80), _MIN_BLOCK_SIZE, _MAX_BLOCK_SIZE,
    )
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    device = grad_output.device
    out = torch.empty(numel, dtype=grad_output.dtype, device=device)

    with torch_device_fn.device(device):
        if n_blocks <= _MAX_REREAD_BLOCKS:
            # Single launch: each program re-reads earlier mask blocks for its
            # exclusive offset, scatters its block, and the last program
            # zero-fills the disjoint tail [total, numel).
            _scatter_reread_kernel[(n_blocks,)](
                grad_output.ravel(), mask.ravel(), out, N, numel, n_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # Triton count kernel -> k on host, then torch_npu's native
            # nonzero via the .out overload (gems does not override
            # aten::nonzero.out — the plain torch.nonzero would dispatch to
            # gems' slow cumsum+scatter, ~32ms @16M), then one Triton flip
            # kernel over the compact range (gather load + affine stores).
            counts = torch.empty(n_blocks, dtype=torch.int64, device=device)
            _count_kernel[(n_blocks,)](
                mask.ravel(), counts, N, BLOCK_SIZE=BLOCK_SIZE,
            )
            k = int(counts.sum().item())
            if k == 0:
                # Empty nonzero result has no storage — hand the kernel a
                # valid pointer; every gather lane is masked off (j < 0), so
                # it is never read.  The kernel then zero-fills all of out.
                idx = torch.zeros(1, dtype=torch.int64, device=device)
            else:
                idx = torch.empty(k, 1, dtype=torch.int64, device=device)
                torch.ops.aten.nonzero.out(mask.ravel(), out=idx)
            grid = triton.cdiv(numel, BLOCK_SIZE)
            _flip_kernel[(grid,)](
                grad_output.ravel(), idx.view(-1), out, k, numel,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out.view(sizes)
