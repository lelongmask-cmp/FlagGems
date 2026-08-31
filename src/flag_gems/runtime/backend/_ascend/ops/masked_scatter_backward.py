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
#   Semantics: out = cat([grad[mask], zeros]).view(sizes)
#
#   main path (#blocks ≤ 128): ONE kernel does everything — each program
#                            re-reads earlier mask blocks (dense vector
#                            loads, torch-written → no sync needed) for its
#                            exclusive offset, scatters its block, and the
#                            last program zero-fills the disjoint tail
#                            [total, numel).
#
#   large N   (#blocks > 128): count(+dense zero-fill of the whole output)
#                            → stream sync (triton kernels writing the same
#                            buffer must not overlap; stream-level sync is
#                            ~7µs vs ~80µs device-wide) → device-side scan
#                            → stream sync → scatter.
#
# Measured platform costs (910B4, triton-ascend 3.2.0):
#   kernel launch ~57µs device time, stream sync ~7µs, device-wide sync ~80µs,
#   scattered store ~80-200ns/lane (dense stores are HBM-bandwidth fast).
#   These fixed costs dominate everything below ~1M elements, so the number
#   of launches/syncs is the primary knob for small/medium shapes.
# ---------------------------------------------------------------------------

_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 1024
_MAX_SCAN_BLOCK = 4096
_MAX_REREAD_BLOCKS = 128
_REREAD_CHUNK = 4096  # (must be restated literally inside jit kernels)


def _stream_sync():
    """Wait for all prior work on the current stream (~7µs on 910B)."""
    torch.npu.current_stream().synchronize()


@libentry()
@triton.jit(do_not_specialize=["N", "numel"])
def _count_zerofill_kernel(
    mask_ptr, counts_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr,
):
    """Per-CTA True count + dense zero-fill of the whole output buffer.

    The zero-fill is done here (first kernel) rather than in a separate
    launch, and densely — the scatter kernel later overwrites the selected
    prefix, so it never reads the zeros (no read-visibility dependency).
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    tl.store(counts_ptr + pid, tl.sum(mask_val, axis=0))
    # Dense zero-fill of out[0:numel) — covered by grid = cdiv(max(N, numel), BLOCK)
    tl.store(out_ptr + offsets, 0.0, mask=(offsets < numel))


@libentry()
@triton.jit(do_not_specialize=["N", "numel"])
def _scatter_kernel(
    grad_ptr, mask_ptr, offsets_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr,
):
    """Scatter True elements to their compact positions (offsets precomputed)."""
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    grad_val = tl.load(grad_ptr + offsets, mask=m, other=0)
    pos = tl.load(offsets_ptr + pid) + tl.cumsum(mask_val, axis=0) - 1
    tl.store(out_ptr + pos, grad_val, mask=m & (mask_val == 1) & (pos < numel))


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


# ---------------------------------------------------------------------------
# device-side exclusive scan over block counts
# ---------------------------------------------------------------------------


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
        _stream_sync()
        chunk_offsets = torch.empty(n_chunks, dtype=torch.int64, device=device)
        scan_block2 = triton.next_power_of_2(n_chunks)
        _scan_kernel[(1,)](
            chunk_totals, chunk_offsets, n_chunks, BLOCK_SIZE=scan_block2,
        )
        _stream_sync()
        _add_offsets_kernel[(n_chunks,)](
            part_sums, chunk_offsets, n_elems, CHUNK_SIZE=_MAX_SCAN_BLOCK,
        )
    return part_sums


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
            # count(+zero-fill) → stream sync → device scan → stream sync → scatter.
            counts = torch.empty(n_blocks, dtype=torch.int64, device=device)
            # Grid covers the larger of N (counting) and numel (zero-fill) so
            # the whole output is zeroed densely even when numel > N.
            grid_c = triton.cdiv(max(N, numel), BLOCK_SIZE)
            _count_zerofill_kernel[(grid_c,)](
                mask.ravel(), counts, out, N, numel, BLOCK_SIZE=BLOCK_SIZE,
            )
            _stream_sync()
            offsets = _exclusive_scan(counts, n_blocks, device)
            _stream_sync()
            _scatter_kernel[(n_blocks,)](
                grad_output.ravel(), mask.ravel(), offsets, out, N, numel,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out.view(sizes)
