/**
 * _unsafe_index_put C++ wrapper v2.
 *
 * Covers ALL parameter forms in C++ (no Python fallback):
 *   1. Bool/int8 masks → expanded via at::nonzero()
 *   2. None indices → filled with at::arange()
 *   3. Missing dims → padded with arange to cover all self dims
 *   4. 2D grid kernel launch → eliminates expensive suffix_numel division
 *   5. Backend-agnostic DMA copy via aten::copy_ redispatch
 *
 * Kernel: triton_src/unsafe_index_put_kernel.py (v2, 2D grid, up to 6 indices).
 */
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <tuple>
#include <vector>

#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
namespace {

  using namespace triton_jit;

  static constexpr int kMaxNdim = 6;

  // ---------------------------------------------------------------------------
  // Host-side utilities
  // ---------------------------------------------------------------------------

  std::vector<int64_t> broadcast_shapes(const std::vector<std::vector<int64_t>>& shapes) {
    if (shapes.empty()) return {};
    int64_t ndim = 0;
    for (auto& s : shapes) ndim = std::max(ndim, static_cast<int64_t>(s.size()));
    std::vector<int64_t> out(ndim, 1);
    for (auto& s : shapes) {
      int64_t pad = ndim - s.size();
      for (int64_t i = 0; i < static_cast<int64_t>(s.size()); i++) {
        int64_t j = pad + i;
        if (s[i] != 1) {
          if (out[j] == 1)
            out[j] = s[i];
          else if (out[j] != s[i])
            TORCH_CHECK(false, "shape mismatch in broadcast_shapes");
        }
      }
    }
    return out;
  }

  std::vector<int64_t> broadcast_strides(const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& target_shape) {
    int64_t ndim = target_shape.size();
    int64_t pad = ndim - shape.size();
    std::vector<int64_t> out(ndim, 0);
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); i++) {
      int64_t j = pad + i;
      if (shape[i] == 1 && target_shape[j] != 1) {
        out[j] = 0;
      } else {
        out[j] = stride[i];
      }
    }
    return out;
  }

  std::vector<int64_t> trailing_divisors(const std::vector<int64_t>& shape) {
    int64_t ndim = shape.size();
    std::vector<int64_t> div(ndim, 1);
    int64_t acc = 1;
    for (int64_t i = ndim - 1; i >= 0; i--) {
      div[i] = acc;
      acc *= shape[i];
    }
    return div;
  }

  template <typename T>
  std::vector<T> pad_vec(const std::vector<T>& seq, int64_t n, T fill) {
    std::vector<T> out = seq;
    out.resize(n, fill);
    return out;
  }

  std::vector<std::vector<int64_t>> pad_2d(const std::vector<std::vector<int64_t>>& arr,
                                           int64_t rows,
                                           int64_t cols,
                                           int64_t fill) {
    std::vector<std::vector<int64_t>> out = arr;
    out.resize(rows, std::vector<int64_t>(cols, fill));
    for (auto& row : out) row.resize(cols, fill);
    return out;
  }

  int64_t volume(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  }

  void heuristic_2d_blocks(int64_t idx_numel, int64_t suffix_numel, int64_t& block_idx, int64_t& block_suf) {
    // Triton requires tl.arange range to be power-of-2.
    auto nearest_pow2 = [](int64_t x, int64_t cap) -> int64_t {
      int64_t v = 1;
      while (v * 2 <= x && v * 2 <= cap) v *= 2;
      return std::max(int64_t(1), std::min(v, cap));
    };
    auto floor_pow2 = [](int64_t x) -> int64_t {
      if (x <= 1) return 1;
      int64_t v = 1;
      while (v * 2 <= x) v *= 2;
      return v;
    };

    constexpr int64_t kTarget = 256;

    if (suffix_numel <= 32) {
      // Small suffix: minimize grid_y (virtually 1D) to reduce launch overhead.
      block_suf = floor_pow2(std::max(int64_t(1), suffix_numel));
      block_idx = floor_pow2(std::max(int64_t(1), std::min(kTarget / block_suf, idx_numel)));
    } else if (suffix_numel >= idx_numel * 4) {
      // Large suffix relative to idx: benefit from 2D grid.
      block_idx = 1;
      block_suf = nearest_pow2(suffix_numel, 256);
    } else if (idx_numel >= suffix_numel * 4) {
      block_suf = std::max(int64_t(1), nearest_pow2(suffix_numel, 256));
      block_idx = nearest_pow2(idx_numel, kTarget / block_suf);
    } else {
      int64_t ratio = idx_numel / std::max(int64_t(1), suffix_numel);
      if (ratio >= 16) {
        block_idx = 32;
        block_suf = 8;
      } else if (ratio >= 4) {
        block_idx = 16;
        block_suf = 16;
      } else {
        block_idx = 8;
        block_suf = 32;
      }
    }
    block_idx = floor_pow2(std::max(int64_t(1), std::min(block_idx, idx_numel)));
    block_suf = floor_pow2(std::max(int64_t(1), std::min(block_suf, suffix_numel)));
  }

  // ---------------------------------------------------------------------------
  // Index preprocessing: handle bool/None/padding in C++
  // ---------------------------------------------------------------------------

  /// Expand a bool/byte mask into LongTensor indices via at::nonzero().
  /// Returns one LongTensor per masked dimension.
  std::vector<at::Tensor> expand_bool_mask(const at::Tensor& mask) {
    auto nonzero = at::nonzero(mask);  // shape (K, ndim)
    std::vector<at::Tensor> result;
    int64_t ndim = nonzero.size(1);
    result.reserve(ndim);
    for (int64_t d = 0; d < ndim; d++) {
      result.push_back(nonzero.select(1, d).contiguous());
    }
    return result;
  }

  /// Preprocess indices: expand bool masks, convert to Long, ensure contiguous.
  std::vector<at::Tensor> preprocess_indices(const c10::List<std::optional<at::Tensor>>& indices) {
    std::vector<at::Tensor> result;
    for (int64_t i = 0; i < indices.size(); i++) {
      TORCH_CHECK(indices[i].has_value(), "_unsafe_index_put does not accept None indices");
      auto dt = indices[i].value().dtype();
      if (dt == at::kBool || dt == at::kByte) {
        auto splits = expand_bool_mask(indices[i].value());
        for (auto& t : splits) result.push_back(t);
      } else {
        auto t = indices[i].value();
        if (t.dtype() != at::kLong) t = t.to(at::kLong);
        result.push_back(t.contiguous());
      }
    }

    TORCH_CHECK(!result.empty(), "at least one index tensor required");
    TORCH_CHECK(static_cast<int64_t>(result.size()) <= kMaxNdim,
                "too many index tensors (max ",
                kMaxNdim,
                ")");
    return result;
  }

  // ---------------------------------------------------------------------------
  // Main kernel launcher
  // ---------------------------------------------------------------------------

  at::Tensor unsafe_index_put_impl(const at::Tensor& self,
                                   const std::vector<at::Tensor>& idx_tensors,
                                   const at::Tensor& values,
                                   bool accumulate) {
    int64_t m = idx_tensors.size();
    int64_t suf_ndim = self.dim() - m;
    TORCH_CHECK(suf_ndim >= 0 && suf_ndim <= kMaxNdim, "suffix ndim out of range: ", suf_ndim);

    // Collect index shapes/strides
    std::vector<std::vector<int64_t>> idx_shapes;
    idx_shapes.reserve(m);
    std::vector<std::vector<int64_t>> idx_strides;
    idx_strides.reserve(m);

    for (int64_t i = 0; i < m; i++) {
      const auto& t = idx_tensors[i];
      std::vector<int64_t> sh(t.sizes().begin(), t.sizes().end());
      std::vector<int64_t> st(t.strides().begin(), t.strides().end());
      idx_shapes.push_back(sh);
      idx_strides.push_back(st);
    }

    // Broadcast index shape
    std::vector<int64_t> idx_shape = broadcast_shapes(idx_shapes);
    int64_t idx_ndim = idx_shape.size();
    TORCH_CHECK(idx_ndim <= kMaxNdim, "index space rank too large: ", idx_ndim);

    // Suffix shape
    std::vector<int64_t> suffix_shape(self.sizes().begin() + m, self.sizes().end());

    int64_t idx_numel = volume(idx_shape);
    int64_t suffix_numel = volume(suffix_shape);
    int64_t N = idx_numel * suffix_numel;

    // ---- Output tensor: empty_like + backend-agnostic copy ----
    auto out = at::empty_like(self, self.options());
    {
      static auto copy_op = c10::Dispatcher::singleton()
                                .findSchemaOrThrow("aten::copy_", "")
                                .typed<at::Tensor&(at::Tensor&, const at::Tensor&, bool)>();
      constexpr c10::DispatchKeySet fallback_keyset(
          c10::DispatchKeySet(c10::DispatchKey::CompositeExplicitAutograd));
      copy_op.redispatch(fallback_keyset, out, self, /*non_blocking=*/false);
    }

    if (N == 0) return out;

    // Accumulate for dtypes without native Triton atomic_add runs a
    // three-kernel scheme with a widened scratch buffer, mirroring PyTorch's
    // opmath_t semantics (accumulate in the wider type, single cast on
    // writeback) without radix sort or CAS flags:
    //   fp16/bf16          → fp32 scratch (atomic fp32 add, one rounding)
    //   int8/int16/uint8   → int32 scratch (lossless; wrap-around on cast back
    //   bool                    matches PyTorch's static_cast semantics)
    //   prologue: seed scratch slots with cast(orig)  (idempotent stores)
    //   main:     atomic_add cast deltas into scratch
    //   epilogue: out = cast(scratch)                 (idempotent stores)
    // The epilogue must not read orig back from `out`: under duplicate target
    // slots that read-modify-write races with other programs' stores and
    // double-counts deltas. Seeding in the prologue keeps every phase either
    // atomic or idempotent.
    at::ScalarType st = self.scalar_type();
    at::ScalarType scratch_dtype;
    bool use_scratch = false;
    if (accumulate) {
      if (st == at::kHalf || st == at::kBFloat16) {
        scratch_dtype = at::kFloat;
        use_scratch = true;
      } else if (st == at::kChar || st == at::kShort || st == at::kByte || st == at::kBool) {
        scratch_dtype = at::kInt;
        use_scratch = true;
      }
    }
    at::Tensor scratch;
    if (use_scratch) {
      // scratch is indexed by the same element offsets used for `out`, so size
      // it to cover out's maximum reachable offset + 1.
      int64_t max_off = 0;
      for (int64_t d = 0; d < out.dim(); d++) {
        if (out.size(d) > 0) max_off += (out.size(d) - 1) * out.stride(d);
      }
      scratch = at::empty({max_off + 1}, self.options().dtype(scratch_dtype));
    }

    // ---- Compute kernel parameters (padded to kMaxNdim) ----
    // Tensor strides in broadcast idx space
    std::vector<std::vector<int64_t>> tensor_strides;
    for (int64_t i = 0; i < m; i++) {
      tensor_strides.push_back(broadcast_strides(idx_shapes[i], idx_strides[i], idx_shape));
    }
    auto tensor_strides_2d = pad_2d(tensor_strides, kMaxNdim, kMaxNdim, int64_t(0));

    // Self advanced strides/sizes (first m dims)
    std::vector<int64_t> self_adv_stride(kMaxNdim, 0);
    std::vector<int64_t> self_adv_size(kMaxNdim, 1);
    for (int64_t d = 0; d < m; d++) {
      self_adv_stride[d] = self.stride(d);
      self_adv_size[d] = self.size(d);
    }

    // Values strides in broadcast target space
    std::vector<int64_t> val_target_shape = idx_shape;
    val_target_shape.insert(val_target_shape.end(), suffix_shape.begin(), suffix_shape.end());
    std::vector<int64_t> val_shape(values.sizes().begin(), values.sizes().end());
    std::vector<int64_t> val_stride_vec(values.strides().begin(), values.strides().end());
    auto val_strides_full = broadcast_strides(val_shape, val_stride_vec, val_target_shape);
    auto val_adv_stride =
        pad_vec(std::vector<int64_t>(val_strides_full.begin(), val_strides_full.begin() + idx_ndim),
                kMaxNdim,
                int64_t(0));
    auto val_suf_stride =
        pad_vec(std::vector<int64_t>(val_strides_full.begin() + idx_ndim, val_strides_full.end()),
                kMaxNdim,
                int64_t(0));

    // Self suffix strides
    std::vector<int64_t> self_suf_stride(kMaxNdim, 0);
    for (int64_t d = 0; d < suf_ndim; d++) {
      self_suf_stride[d] = self.stride(m + d);
    }

    // Divisors
    auto idx_div = pad_vec(trailing_divisors(idx_shape), kMaxNdim, int64_t(1));
    auto suf_div = pad_vec(trailing_divisors(suffix_shape), kMaxNdim, int64_t(1));

    // ---- 2D grid parameters ----
    int64_t block_idx, block_suf;
    heuristic_2d_blocks(idx_numel, suffix_numel, block_idx, block_suf);

    int64_t grid_x = (idx_numel + block_idx - 1) / block_idx;
    int64_t grid_y = (suffix_numel + block_suf - 1) / block_suf;

    // ---- Launch Triton kernel ----
    c10::DeviceGuard guard(out.device());
    auto stream = backend::getCurrentStream();
    auto raw_stream = backend::getRawStream(stream);

    const TritonJITFunction& kernel = TritonJITFunction::get_instance(
        (utils::get_triton_src_path() / "unsafe_index_put_kernel.py").string(),
        "unsafe_index_put_kernel_v2");

    // Pad index tensor list for kernel args (kernel always takes kMaxNdim pointers)
    std::vector<at::Tensor> kernel_idx = idx_tensors;
    while (kernel_idx.size() < static_cast<size_t>(kMaxNdim)) {
      kernel_idx.push_back(kernel_idx.empty() ? at::Tensor() : kernel_idx[0]);
    }

    // scratch arg: pass the real buffer on the scratch path, a dummy otherwise
    const at::Tensor& scratch_arg = use_scratch ? scratch : out;

    // Lambda that launches the scratch prologue/epilogue kernel; both share the
    // same argument layout and only differ in the PROLOGUE constexpr.
    auto launch_scratch_kernel = [&](bool prologue) {
      const TritonJITFunction& skernel = TritonJITFunction::get_instance(
          (utils::get_triton_src_path() / "unsafe_index_put_kernel.py").string(),
          "unsafe_index_put_scratch_kernel");
      skernel(raw_stream,
              static_cast<unsigned int>(grid_x),
              static_cast<unsigned int>(grid_y),
              1,  // grid_z
              4,  // num_warps
              4,  // num_stages
              // output / scratch
              out,
              scratch,
              // index data pointers (kMaxNdim)
              kernel_idx[0],
              kernel_idx[1],
              kernel_idx[2],
              kernel_idx[3],
              kernel_idx[4],
              kernel_idx[5],
              // idx_div (kMaxNdim)
              idx_div[0],
              idx_div[1],
              idx_div[2],
              idx_div[3],
              idx_div[4],
              idx_div[5],
              // tensor_strides (kMaxNdim × kMaxNdim = 36 scalars)
              tensor_strides_2d[0][0],
              tensor_strides_2d[0][1],
              tensor_strides_2d[0][2],
              tensor_strides_2d[0][3],
              tensor_strides_2d[0][4],
              tensor_strides_2d[0][5],
              tensor_strides_2d[1][0],
              tensor_strides_2d[1][1],
              tensor_strides_2d[1][2],
              tensor_strides_2d[1][3],
              tensor_strides_2d[1][4],
              tensor_strides_2d[1][5],
              tensor_strides_2d[2][0],
              tensor_strides_2d[2][1],
              tensor_strides_2d[2][2],
              tensor_strides_2d[2][3],
              tensor_strides_2d[2][4],
              tensor_strides_2d[2][5],
              tensor_strides_2d[3][0],
              tensor_strides_2d[3][1],
              tensor_strides_2d[3][2],
              tensor_strides_2d[3][3],
              tensor_strides_2d[3][4],
              tensor_strides_2d[3][5],
              tensor_strides_2d[4][0],
              tensor_strides_2d[4][1],
              tensor_strides_2d[4][2],
              tensor_strides_2d[4][3],
              tensor_strides_2d[4][4],
              tensor_strides_2d[4][5],
              tensor_strides_2d[5][0],
              tensor_strides_2d[5][1],
              tensor_strides_2d[5][2],
              tensor_strides_2d[5][3],
              tensor_strides_2d[5][4],
              tensor_strides_2d[5][5],
              // self_adv_stride (kMaxNdim)
              self_adv_stride[0],
              self_adv_stride[1],
              self_adv_stride[2],
              self_adv_stride[3],
              self_adv_stride[4],
              self_adv_stride[5],
              // self_adv_size (kMaxNdim)
              self_adv_size[0],
              self_adv_size[1],
              self_adv_size[2],
              self_adv_size[3],
              self_adv_size[4],
              self_adv_size[5],
              // suf_div (kMaxNdim)
              suf_div[0],
              suf_div[1],
              suf_div[2],
              suf_div[3],
              suf_div[4],
              suf_div[5],
              // self_suf_stride (kMaxNdim)
              self_suf_stride[0],
              self_suf_stride[1],
              self_suf_stride[2],
              self_suf_stride[3],
              self_suf_stride[4],
              self_suf_stride[5],
              // meta
              idx_numel,
              suffix_numel,
              N,
              // constexpr params
              static_cast<int32_t>(m),
              static_cast<int32_t>(idx_ndim),
              static_cast<int32_t>(suf_ndim),
              prologue,
              static_cast<int32_t>(block_idx),
              static_cast<int32_t>(block_suf));
    };

    // Prologue: seed the fp32 scratch slots with fp32(orig).
    if (use_scratch) {
      launch_scratch_kernel(/*prologue=*/true);
    }

    // Build the massive argument list for the 2D grid kernel.
    // Order must match unsafe_index_put_kernel_v2 exactly.
    kernel(raw_stream,
           static_cast<unsigned int>(grid_x),
           static_cast<unsigned int>(grid_y),
           1,  // grid_z
           4,  // num_warps
           4,  // num_stages
           // outputs / values / accumulate scratch
           out,
           values,
           scratch_arg,
           // index data pointers (kMaxNdim)
           kernel_idx[0],
           kernel_idx[1],
           kernel_idx[2],
           kernel_idx[3],
           kernel_idx[4],
           kernel_idx[5],
           // idx_div (kMaxNdim)
           idx_div[0],
           idx_div[1],
           idx_div[2],
           idx_div[3],
           idx_div[4],
           idx_div[5],
           // tensor_strides (kMaxNdim × kMaxNdim = 36 scalars)
           tensor_strides_2d[0][0],
           tensor_strides_2d[0][1],
           tensor_strides_2d[0][2],
           tensor_strides_2d[0][3],
           tensor_strides_2d[0][4],
           tensor_strides_2d[0][5],
           tensor_strides_2d[1][0],
           tensor_strides_2d[1][1],
           tensor_strides_2d[1][2],
           tensor_strides_2d[1][3],
           tensor_strides_2d[1][4],
           tensor_strides_2d[1][5],
           tensor_strides_2d[2][0],
           tensor_strides_2d[2][1],
           tensor_strides_2d[2][2],
           tensor_strides_2d[2][3],
           tensor_strides_2d[2][4],
           tensor_strides_2d[2][5],
           tensor_strides_2d[3][0],
           tensor_strides_2d[3][1],
           tensor_strides_2d[3][2],
           tensor_strides_2d[3][3],
           tensor_strides_2d[3][4],
           tensor_strides_2d[3][5],
           tensor_strides_2d[4][0],
           tensor_strides_2d[4][1],
           tensor_strides_2d[4][2],
           tensor_strides_2d[4][3],
           tensor_strides_2d[4][4],
           tensor_strides_2d[4][5],
           tensor_strides_2d[5][0],
           tensor_strides_2d[5][1],
           tensor_strides_2d[5][2],
           tensor_strides_2d[5][3],
           tensor_strides_2d[5][4],
           tensor_strides_2d[5][5],
           // val_adv_stride (kMaxNdim)
           val_adv_stride[0],
           val_adv_stride[1],
           val_adv_stride[2],
           val_adv_stride[3],
           val_adv_stride[4],
           val_adv_stride[5],
           // self_adv_stride (kMaxNdim)
           self_adv_stride[0],
           self_adv_stride[1],
           self_adv_stride[2],
           self_adv_stride[3],
           self_adv_stride[4],
           self_adv_stride[5],
           // self_adv_size (kMaxNdim)
           self_adv_size[0],
           self_adv_size[1],
           self_adv_size[2],
           self_adv_size[3],
           self_adv_size[4],
           self_adv_size[5],
           // suf_div (kMaxNdim)
           suf_div[0],
           suf_div[1],
           suf_div[2],
           suf_div[3],
           suf_div[4],
           suf_div[5],
           // self_suf_stride (kMaxNdim)
           self_suf_stride[0],
           self_suf_stride[1],
           self_suf_stride[2],
           self_suf_stride[3],
           self_suf_stride[4],
           self_suf_stride[5],
           // val_suf_stride (kMaxNdim)
           val_suf_stride[0],
           val_suf_stride[1],
           val_suf_stride[2],
           val_suf_stride[3],
           val_suf_stride[4],
           val_suf_stride[5],
           // meta
           idx_numel,
           suffix_numel,
           N,
           // constexpr params
           static_cast<int32_t>(m),
           static_cast<int32_t>(idx_ndim),
           static_cast<int32_t>(suf_ndim),
           accumulate,
           use_scratch,
           static_cast<int32_t>(block_idx),
           static_cast<int32_t>(block_suf));

    // Epilogue: out = cast(scratch) with a single rounding.
    if (use_scratch) {
      launch_scratch_kernel(/*prologue=*/false);
    }

    return out;
  }

}  // namespace

// ---------------------------------------------------------------------------
// Operator dispatch: all parameter forms handled in C++
// ---------------------------------------------------------------------------
at::Tensor unsafe_index_put_cpp(const at::Tensor& self,
                                const c10::List<std::optional<at::Tensor>>& indices,
                                const at::Tensor& values,
                                bool accumulate) {
  // Preprocess: bool→int, int→long, contiguous
  auto processed = preprocess_indices(indices);
  // fp16/bf16 accumulate is handled inside impl via the fp32-scratch
  // three-kernel scheme (opmath_t-equivalent, sort-free).
  return unsafe_index_put_impl(self, processed, values, accumulate);
}

}  // namespace flag_gems
