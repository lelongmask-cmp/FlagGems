"""
_unsafe_index_put Triton kernel v2 — 2D grid for C++ via TritonJITFunction.

Key changes from v1:
- 2D grid: program_id(0)→idx_pos, program_id(1)→suf_pos. Eliminates expensive
  integer division by suffix_numel (the dominant cost on large-suffix shapes).
- Supports up to 6 index tensors and 6 suffix dims (was 4).
- Direct @triton.jit, no @libentry().
"""
import triton
import triton.language as tl


@triton.jit
def unsafe_index_put_kernel_v2(
    out_ptr,
    values_ptr,
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
    USE_CAS: tl.constexpr,
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
        if USE_CAS:
            # CAS-based atomic add — universal dtype support.
            # tl.atomic_add has limited dtype support (no int8/uint8/int16,
            # and float16/bfloat16 produce wrong results on some hardware).
            #
            # tl.atomic_cas does not support a mask parameter, so the loop
            # runs for all elements in the block. Safe by design:
            #  - Tail elements (mask=False): redirected to element 0,
            #    CAS with cmp=val=0 → no-op (no modification).
            #  - Successful lanes: `succeeded` flag keeps `old` unchanged,
            #    subsequent CAS attempts fail harmlessly (cmp != *ptr).
            old = tl.load(out_ptr + self_off, mask=mask, other=0)
            safe_off = tl.where(mask, self_off, tl.zeros_like(self_off))
            # Bitcast the output pointer to uint16 so that tl.atomic_cas
            # performs an exact integer comparison (matching CUDA's native
            # atomicAdd implementation for fp16/bf16).  Otherwise float
            # comparison quirks (-0.0 vs +0.0) cause lost accumulations.
            uint_ptr = out_ptr.to(tl.pointer_type(tl.uint16))
            for _ in range(10):
                # Match CUDA atomicAdd semantics: float32 accumulation
                # rounded to native dtype for the atomic compare-and-swap.
                new_val = (old.to(tl.float32) + v.to(tl.float32)).to(old.dtype)
                old_bits = old.to(tl.uint16, bitcast=True)
                new_bits = new_val.to(tl.uint16, bitcast=True)
                actual_bits = tl.atomic_cas(uint_ptr + safe_off, old_bits, new_bits)
                actual = actual_bits.to(old.dtype, bitcast=True)
                # For lanes where CAS succeeded, zero out v so subsequent
                # iterations are no-ops.  Integer comparison is exact bitwise.
                v = tl.where(actual_bits == old_bits, tl.zeros_like(v), v)
                old = actual
        else:
            tl.atomic_add(out_ptr + self_off, v, mask=mask)
    else:
        tl.store(out_ptr + self_off, v, mask=mask)
