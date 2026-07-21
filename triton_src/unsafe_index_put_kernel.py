"""
_unsafe_index_put Triton kernel v2 — 2D grid for C++ via TritonJITFunction.

Key changes from v1:
- 2D grid: program_id(0)→idx_pos, program_id(1)→suf_pos. Eliminates expensive
  integer division by suffix_numel (the dominant cost on large-suffix shapes).
- Supports up to 6 index tensors and 6 suffix dims (was 4).
- Direct @triton.jit, no @libentry().
- fp16/bf16 accumulate: three-kernel scheme with an fp32 scratch buffer,
  matching PyTorch's opmath_t semantics (accumulate in fp32, single rounding
  on writeback) without radix-sort or CAS/flag overhead.

Kernel 0 (unsafe_index_put_scratch_kernel, PROLOGUE=True):
  Stores 0.0f into the fp32 scratch at every touched offset. Duplicate stores
  are harmless (all write the same value). Stream ordering guarantees this
  completes before accumulation starts.

Kernel 1 (unsafe_index_put_kernel_v2, USE_SCRATCH=True):
  atomic_add's each fp32-cast value into scratch. Pure-delta accumulation in
  fp32; no read-modify-write of the output.

Kernel 2 (unsafe_index_put_scratch_kernel, PROLOGUE=False):
  Reads fp32 scratch and the fp16/bf16 original (already cloned into out),
  adds them in fp32, casts once to the output dtype, and stores.
"""
import triton
import triton.language as tl


@triton.jit
def unsafe_index_put_kernel_v2(
    out_ptr,
    values_ptr,
    scratch_ptr,
    idx0_ptr, idx1_ptr, idx2_ptr, idx3_ptr, idx4_ptr, idx5_ptr,
    idx_div0, idx_div1, idx_div2, idx_div3, idx_div4, idx_div5,
    ts_0_0, ts_0_1, ts_0_2, ts_0_3, ts_0_4, ts_0_5,
    ts_1_0, ts_1_1, ts_1_2, ts_1_3, ts_1_4, ts_1_5,
    ts_2_0, ts_2_1, ts_2_2, ts_2_3, ts_2_4, ts_2_5,
    ts_3_0, ts_3_1, ts_3_2, ts_3_3, ts_3_4, ts_3_5,
    ts_4_0, ts_4_1, ts_4_2, ts_4_3, ts_4_4, ts_4_5,
    ts_5_0, ts_5_1, ts_5_2, ts_5_3, ts_5_4, ts_5_5,
    val_adv0, val_adv1, val_adv2, val_adv3, val_adv4, val_adv5,
    self_adv_stride0, self_adv_stride1, self_adv_stride2,
    self_adv_stride3, self_adv_stride4, self_adv_stride5,
    self_adv_size0, self_adv_size1, self_adv_size2,
    self_adv_size3, self_adv_size4, self_adv_size5,
    suf_div0, suf_div1, suf_div2, suf_div3, suf_div4, suf_div5,
    self_suf_stride0, self_suf_stride1, self_suf_stride2,
    self_suf_stride3, self_suf_stride4, self_suf_stride5,
    val_suf_stride0, val_suf_stride1, val_suf_stride2,
    val_suf_stride3, val_suf_stride4, val_suf_stride5,
    idx_numel,
    suffix_numel,
    N,
    M: tl.constexpr,
    IDX_NDIM: tl.constexpr,
    SUF_NDIM: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    USE_SCRATCH: tl.constexpr,
    BLOCK_IDX: tl.constexpr,
    BLOCK_SUF: tl.constexpr,
):
    """
    2D grid kernel.

    Grid: (cdiv(idx_numel, BLOCK_IDX), cdiv(suffix_numel, BLOCK_SUF)).
    Each block handles BLOCK_IDX index positions × BLOCK_SUF suffix positions.
    program_id(0) → idx position range (no expensive division by suffix_numel!).
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    idx_off = pid0 * BLOCK_IDX + tl.arange(0, BLOCK_IDX)[:, None]   # (BI, 1)
    suf_off = pid1 * BLOCK_SUF + tl.arange(0, BLOCK_SUF)[None, :]   # (1, BS)

    mask_idx = idx_off < idx_numel
    mask_suf = suf_off < suffix_numel
    mask = mask_idx & mask_suf  # (BI, BS)

    val_off = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    self_off = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    toff0 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff1 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff2 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff3 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff4 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff5 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    rem_idx = idx_off

    # ---- index-space coordinate decomposition ----
    if IDX_NDIM >= 1:
        c0 = rem_idx // idx_div0
        rem_idx = rem_idx % idx_div0
        val_off += c0 * val_adv0
        if M >= 1: toff0 += c0 * ts_0_0
        if M >= 2: toff1 += c0 * ts_1_0
        if M >= 3: toff2 += c0 * ts_2_0
        if M >= 4: toff3 += c0 * ts_3_0
        if M >= 5: toff4 += c0 * ts_4_0
        if M >= 6: toff5 += c0 * ts_5_0

    if IDX_NDIM >= 2:
        c1 = rem_idx // idx_div1
        rem_idx = rem_idx % idx_div1
        val_off += c1 * val_adv1
        if M >= 1: toff0 += c1 * ts_0_1
        if M >= 2: toff1 += c1 * ts_1_1
        if M >= 3: toff2 += c1 * ts_2_1
        if M >= 4: toff3 += c1 * ts_3_1
        if M >= 5: toff4 += c1 * ts_4_1
        if M >= 6: toff5 += c1 * ts_5_1

    if IDX_NDIM >= 3:
        c2 = rem_idx // idx_div2
        rem_idx = rem_idx % idx_div2
        val_off += c2 * val_adv2
        if M >= 1: toff0 += c2 * ts_0_2
        if M >= 2: toff1 += c2 * ts_1_2
        if M >= 3: toff2 += c2 * ts_2_2
        if M >= 4: toff3 += c2 * ts_3_2
        if M >= 5: toff4 += c2 * ts_4_2
        if M >= 6: toff5 += c2 * ts_5_2

    if IDX_NDIM >= 4:
        c3 = rem_idx // idx_div3
        rem_idx = rem_idx % idx_div3
        val_off += c3 * val_adv3
        if M >= 1: toff0 += c3 * ts_0_3
        if M >= 2: toff1 += c3 * ts_1_3
        if M >= 3: toff2 += c3 * ts_2_3
        if M >= 4: toff3 += c3 * ts_3_3
        if M >= 5: toff4 += c3 * ts_4_3
        if M >= 6: toff5 += c3 * ts_5_3

    if IDX_NDIM >= 5:
        c4 = rem_idx // idx_div4
        rem_idx = rem_idx % idx_div4
        val_off += c4 * val_adv4
        if M >= 1: toff0 += c4 * ts_0_4
        if M >= 2: toff1 += c4 * ts_1_4
        if M >= 3: toff2 += c4 * ts_2_4
        if M >= 4: toff3 += c4 * ts_3_4
        if M >= 5: toff4 += c4 * ts_4_4
        if M >= 6: toff5 += c4 * ts_5_4

    if IDX_NDIM >= 6:
        c5 = rem_idx // idx_div5
        rem_idx = rem_idx % idx_div5
        val_off += c5 * val_adv5
        if M >= 1: toff0 += c5 * ts_0_5
        if M >= 2: toff1 += c5 * ts_1_5
        if M >= 3: toff2 += c5 * ts_2_5
        if M >= 4: toff3 += c5 * ts_3_5
        if M >= 5: toff4 += c5 * ts_4_5
        if M >= 6: toff5 += c5 * ts_5_5

    # ---- load index values ----
    if M >= 1:
        idx0_ptr = idx0_ptr.to(tl.pointer_type(tl.int64))
    if M >= 2:
        idx1_ptr = idx1_ptr.to(tl.pointer_type(tl.int64))
    if M >= 3:
        idx2_ptr = idx2_ptr.to(tl.pointer_type(tl.int64))
    if M >= 4:
        idx3_ptr = idx3_ptr.to(tl.pointer_type(tl.int64))
    if M >= 5:
        idx4_ptr = idx4_ptr.to(tl.pointer_type(tl.int64))
    if M >= 6:
        idx5_ptr = idx5_ptr.to(tl.pointer_type(tl.int64))

    if M >= 1:
        ind = tl.load(idx0_ptr + toff0, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size0, ind)
        self_off += ind * self_adv_stride0
    if M >= 2:
        ind = tl.load(idx1_ptr + toff1, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size1, ind)
        self_off += ind * self_adv_stride1
    if M >= 3:
        ind = tl.load(idx2_ptr + toff2, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size2, ind)
        self_off += ind * self_adv_stride2
    if M >= 4:
        ind = tl.load(idx3_ptr + toff3, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size3, ind)
        self_off += ind * self_adv_stride3
    if M >= 5:
        ind = tl.load(idx4_ptr + toff4, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size4, ind)
        self_off += ind * self_adv_stride4
    if M >= 6:
        ind = tl.load(idx5_ptr + toff5, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size5, ind)
        self_off += ind * self_adv_stride5

    # ---- suffix coordinate decomposition ----
    rem_suf = suf_off
    if SUF_NDIM >= 1:
        cs0 = rem_suf // suf_div0
        rem_suf = rem_suf % suf_div0
        self_off += cs0 * self_suf_stride0
        val_off += cs0 * val_suf_stride0
    if SUF_NDIM >= 2:
        cs1 = rem_suf // suf_div1
        rem_suf = rem_suf % suf_div1
        self_off += cs1 * self_suf_stride1
        val_off += cs1 * val_suf_stride1
    if SUF_NDIM >= 3:
        cs2 = rem_suf // suf_div2
        rem_suf = rem_suf % suf_div2
        self_off += cs2 * self_suf_stride2
        val_off += cs2 * val_suf_stride2
    if SUF_NDIM >= 4:
        cs3 = rem_suf // suf_div3
        rem_suf = rem_suf % suf_div3
        self_off += cs3 * self_suf_stride3
        val_off += cs3 * val_suf_stride3
    if SUF_NDIM >= 5:
        cs4 = rem_suf // suf_div4
        rem_suf = rem_suf % suf_div4
        self_off += cs4 * self_suf_stride4
        val_off += cs4 * val_suf_stride4
    if SUF_NDIM >= 6:
        cs5 = rem_suf // suf_div5
        rem_suf = rem_suf % suf_div5
        self_off += cs5 * self_suf_stride5
        val_off += cs5 * val_suf_stride5

    # ---- load and store/accumulate ----
    v = tl.load(values_ptr + val_off, mask=mask, other=0.0)
    if ACCUMULATE:
        if USE_SCRATCH:
            # Scratch-based accumulate for outputs whose dtype lacks native
            # atomic_add (fp16/bf16 → fp32 scratch; int8/int16/uint8/bool →
            # int32 scratch). Scratch slots were seeded by the prologue; here
            # we only add the cast delta. Lossless for all supported dtypes.
            tl.atomic_add(scratch_ptr + self_off,
                          v.to(scratch_ptr.dtype.element_ty), mask=mask)
        else:
            tl.atomic_add(out_ptr + self_off, v, mask=mask)
    else:
        tl.store(out_ptr + self_off, v, mask=mask)


@triton.jit
def unsafe_index_put_scratch_kernel(
    out_ptr,
    scratch_ptr,
    idx0_ptr, idx1_ptr, idx2_ptr, idx3_ptr, idx4_ptr, idx5_ptr,
    idx_div0, idx_div1, idx_div2, idx_div3, idx_div4, idx_div5,
    ts_0_0, ts_0_1, ts_0_2, ts_0_3, ts_0_4, ts_0_5,
    ts_1_0, ts_1_1, ts_1_2, ts_1_3, ts_1_4, ts_1_5,
    ts_2_0, ts_2_1, ts_2_2, ts_2_3, ts_2_4, ts_2_5,
    ts_3_0, ts_3_1, ts_3_2, ts_3_3, ts_3_4, ts_3_5,
    ts_4_0, ts_4_1, ts_4_2, ts_4_3, ts_4_4, ts_4_5,
    ts_5_0, ts_5_1, ts_5_2, ts_5_3, ts_5_4, ts_5_5,
    self_adv_stride0, self_adv_stride1, self_adv_stride2,
    self_adv_stride3, self_adv_stride4, self_adv_stride5,
    self_adv_size0, self_adv_size1, self_adv_size2,
    self_adv_size3, self_adv_size4, self_adv_size5,
    suf_div0, suf_div1, suf_div2, suf_div3, suf_div4, suf_div5,
    self_suf_stride0, self_suf_stride1, self_suf_stride2,
    self_suf_stride3, self_suf_stride4, self_suf_stride5,
    idx_numel,
    suffix_numel,
    N,
    M: tl.constexpr,
    IDX_NDIM: tl.constexpr,
    SUF_NDIM: tl.constexpr,
    PROLOGUE: tl.constexpr,
    BLOCK_IDX: tl.constexpr,
    BLOCK_SUF: tl.constexpr,
):
    """
    Scratch prologue/epilogue for the fp32-scratch accumulate path (fp16/bf16).
    Recomputes the touched offsets exactly as kernel_v2 does.

    PROLOGUE=True:  seeds scratch slots with fp32(orig) read from the cloned
                    output (idempotent under duplicate slots).
    PROLOGUE=False: stores cast(scratch) into out (idempotent too).
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    idx_off = pid0 * BLOCK_IDX + tl.arange(0, BLOCK_IDX)[:, None]   # (BI, 1)
    suf_off = pid1 * BLOCK_SUF + tl.arange(0, BLOCK_SUF)[None, :]   # (1, BS)

    mask_idx = idx_off < idx_numel
    mask_suf = suf_off < suffix_numel
    mask = mask_idx & mask_suf  # (BI, BS)

    self_off = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    toff0 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff1 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff2 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff3 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff4 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff5 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    rem_idx = idx_off

    # ---- index-space coordinate decomposition ----
    if IDX_NDIM >= 1:
        c0 = rem_idx // idx_div0
        rem_idx = rem_idx % idx_div0
        if M >= 1: toff0 += c0 * ts_0_0
        if M >= 2: toff1 += c0 * ts_1_0
        if M >= 3: toff2 += c0 * ts_2_0
        if M >= 4: toff3 += c0 * ts_3_0
        if M >= 5: toff4 += c0 * ts_4_0
        if M >= 6: toff5 += c0 * ts_5_0

    if IDX_NDIM >= 2:
        c1 = rem_idx // idx_div1
        rem_idx = rem_idx % idx_div1
        if M >= 1: toff0 += c1 * ts_0_1
        if M >= 2: toff1 += c1 * ts_1_1
        if M >= 3: toff2 += c1 * ts_2_1
        if M >= 4: toff3 += c1 * ts_3_1
        if M >= 5: toff4 += c1 * ts_4_1
        if M >= 6: toff5 += c1 * ts_5_1

    if IDX_NDIM >= 3:
        c2 = rem_idx // idx_div2
        rem_idx = rem_idx % idx_div2
        if M >= 1: toff0 += c2 * ts_0_2
        if M >= 2: toff1 += c2 * ts_1_2
        if M >= 3: toff2 += c2 * ts_2_2
        if M >= 4: toff3 += c2 * ts_3_2
        if M >= 5: toff4 += c2 * ts_4_2
        if M >= 6: toff5 += c2 * ts_5_2

    if IDX_NDIM >= 4:
        c3 = rem_idx // idx_div3
        rem_idx = rem_idx % idx_div3
        if M >= 1: toff0 += c3 * ts_0_3
        if M >= 2: toff1 += c3 * ts_1_3
        if M >= 3: toff2 += c3 * ts_2_3
        if M >= 4: toff3 += c3 * ts_3_3
        if M >= 5: toff4 += c3 * ts_4_3
        if M >= 6: toff5 += c3 * ts_5_3

    if IDX_NDIM >= 5:
        c4 = rem_idx // idx_div4
        rem_idx = rem_idx % idx_div4
        if M >= 1: toff0 += c4 * ts_0_4
        if M >= 2: toff1 += c4 * ts_1_4
        if M >= 3: toff2 += c4 * ts_2_4
        if M >= 4: toff3 += c4 * ts_3_4
        if M >= 5: toff4 += c4 * ts_4_4
        if M >= 6: toff5 += c4 * ts_5_4

    if IDX_NDIM >= 6:
        c5 = rem_idx // idx_div5
        rem_idx = rem_idx % idx_div5
        if M >= 1: toff0 += c5 * ts_0_5
        if M >= 2: toff1 += c5 * ts_1_5
        if M >= 3: toff2 += c5 * ts_2_5
        if M >= 4: toff3 += c5 * ts_3_5
        if M >= 5: toff4 += c5 * ts_4_5
        if M >= 6: toff5 += c5 * ts_5_5

    # ---- load index values ----
    if M >= 1:
        idx0_ptr = idx0_ptr.to(tl.pointer_type(tl.int64))
    if M >= 2:
        idx1_ptr = idx1_ptr.to(tl.pointer_type(tl.int64))
    if M >= 3:
        idx2_ptr = idx2_ptr.to(tl.pointer_type(tl.int64))
    if M >= 4:
        idx3_ptr = idx3_ptr.to(tl.pointer_type(tl.int64))
    if M >= 5:
        idx4_ptr = idx4_ptr.to(tl.pointer_type(tl.int64))
    if M >= 6:
        idx5_ptr = idx5_ptr.to(tl.pointer_type(tl.int64))

    if M >= 1:
        ind = tl.load(idx0_ptr + toff0, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size0, ind)
        self_off += ind * self_adv_stride0
    if M >= 2:
        ind = tl.load(idx1_ptr + toff1, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size1, ind)
        self_off += ind * self_adv_stride1
    if M >= 3:
        ind = tl.load(idx2_ptr + toff2, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size2, ind)
        self_off += ind * self_adv_stride2
    if M >= 4:
        ind = tl.load(idx3_ptr + toff3, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size3, ind)
        self_off += ind * self_adv_stride3
    if M >= 5:
        ind = tl.load(idx4_ptr + toff4, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size4, ind)
        self_off += ind * self_adv_stride4
    if M >= 6:
        ind = tl.load(idx5_ptr + toff5, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size5, ind)
        self_off += ind * self_adv_stride5

    # ---- suffix coordinate decomposition ----
    rem_suf = suf_off
    if SUF_NDIM >= 1:
        cs0 = rem_suf // suf_div0
        rem_suf = rem_suf % suf_div0
        self_off += cs0 * self_suf_stride0
    if SUF_NDIM >= 2:
        cs1 = rem_suf // suf_div1
        rem_suf = rem_suf % suf_div1
        self_off += cs1 * self_suf_stride1
    if SUF_NDIM >= 3:
        cs2 = rem_suf // suf_div2
        rem_suf = rem_suf % suf_div2
        self_off += cs2 * self_suf_stride2
    if SUF_NDIM >= 4:
        cs3 = rem_suf // suf_div3
        rem_suf = rem_suf % suf_div3
        self_off += cs3 * self_suf_stride3
    if SUF_NDIM >= 5:
        cs4 = rem_suf // suf_div4
        rem_suf = rem_suf % suf_div4
        self_off += cs4 * self_suf_stride4
    if SUF_NDIM >= 6:
        cs5 = rem_suf // suf_div5
        rem_suf = rem_suf % suf_div5
        self_off += cs5 * self_suf_stride5

    # ---- prologue: seed scratch; epilogue: write final values ----
    # Both phases are race-free under duplicate target slots:
    #   prologue: every writer stores the SAME fp32(orig) value (idempotent).
    #   epilogue: reads only scratch (stable after the main kernel), and every
    #             writer stores the SAME final value (idempotent).
    # No non-atomic read-modify-write of `out` anywhere: the epilogue must NOT
    # read orig from `out`, otherwise one program's store could be observed as
    # another program's "orig", doubling the accumulated deltas.
    if PROLOGUE:
        orig = tl.load(out_ptr + self_off, mask=mask, other=0.0)
        tl.store(scratch_ptr + self_off,
                 orig.to(scratch_ptr.dtype.element_ty), mask=mask)
    else:
        v32 = tl.load(scratch_ptr + self_off, mask=mask, other=0.0)
        tl.store(out_ptr + self_off, v32.to(out_ptr.dtype.element_ty),
                 mask=mask)
