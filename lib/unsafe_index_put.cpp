/**
 * _unsafe_index_put C++ wrapper.
 *
 * 与 Python wrapper 的关键区别:
 *   1. clone 走 cudaMemcpyAsync (DMA), 不经过 FlagGems 的 Triton _copy_kernel
 *   2. 所有 host 端计算 (broadcast shape/stride/padding) 在 C++ 完成, 无 Python 开销
 *   3. 直接通过 TritonJITFunction 启动 kernel, 绕过 @libentry() 和 use_gems()
 */
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <array>
#include <cstring>
#include <numeric>
#include <tuple>
#include <vector>

#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
namespace {

using namespace triton_jit;

// ---------------------------------------------------------------------------
// Host 端工具函数 (C++ 实现, 对应 Python 的 _broadcast_shape /
// _broadcast_strides / _trailing_divisors / _pad / _heuristic_block)
// ---------------------------------------------------------------------------
static constexpr int kMaxNdim = 4;

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
        if (out[j] == 1) out[j] = s[i];
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
      out[j] = 0;  // broadcast dim
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

int64_t heuristic_block(int64_t n) {
  // Choose block size to balance occupancy and per-block work.
  // For small N, use smaller blocks to get more parallelism.
  // For moderate N, use medium blocks.
  // For large N, use larger blocks to reduce grid launch overhead.
  if (n <= 1024) return 128;
  if (n <= 8192) return 256;
  if (n <= 65536) return 512;
  return 1024;
}

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------

at::Tensor unsafe_index_put(const at::Tensor& self,
                            const c10::List<std::optional<at::Tensor>>& indices,
                            const at::Tensor& values,
                            bool accumulate) {
  // ---- 快速路径判定 ----
  if (indices.empty()) {
    TORCH_CHECK(false, "at least one index must be provided");
  }

  int64_t m = indices.size();
  TORCH_CHECK(m <= kMaxNdim, "too many index tensors (max ", kMaxNdim, ")");
  TORCH_CHECK(m <= self.dim(), "too many indices for tensor of dimension ", self.dim());

  // 检查所有索引都是非 None 的张量
  for (int64_t i = 0; i < m; i++) {
    TORCH_CHECK(indices[i].has_value(), "None indices not supported in fast path");
  }

  int64_t suf_ndim = self.dim() - m;
  TORCH_CHECK(suf_ndim <= kMaxNdim, "too many suffix dims (max ", kMaxNdim, ")");

  // 收集 index shapes/strides
  std::vector<std::vector<int64_t>> idx_shapes;
  idx_shapes.reserve(m);
  std::vector<std::vector<int64_t>> idx_strides;
  idx_strides.reserve(m);
  std::vector<at::Tensor> idx_tensors;
  idx_tensors.reserve(m);

  for (int64_t i = 0; i < m; i++) {
    at::Tensor t = indices[i].value();
    if (t.dtype() != at::kLong) t = t.to(at::kLong);
    idx_tensors.push_back(t.contiguous());
    std::vector<int64_t> sh(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> st(t.strides().begin(), t.strides().end());
    idx_shapes.push_back(sh);
    idx_strides.push_back(st);
  }

  // broadcast index shape
  std::vector<int64_t> idx_shape = broadcast_shapes(idx_shapes);
  int64_t idx_ndim = idx_shape.size();
  TORCH_CHECK(idx_ndim <= kMaxNdim, "index space rank too large");

  // suffix shape
  std::vector<int64_t> suffix_shape(self.sizes().begin() + m, self.sizes().end());

  int64_t idx_numel = volume(idx_shape);
  int64_t suffix_numel = volume(suffix_shape);
  int64_t N = idx_numel * suffix_numel;

  // ---- 输出 tensor: empty_like + backend-agnostic copy ----
  // 使用 aten::copy_ redispatch 到 CompositeExplicitAutograd,
  // 绕过 FlagGems dispatch, 直接使用 PyTorch 原生 copy
  // (在 CUDA 上使用 cudaMemcpyAsync, 在 NPU 上使用 aclrtMemcpy, 等等)
  auto out = at::empty_like(self, self.options());
  {
    static auto copy_op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("aten::copy_", "")
            .typed<at::Tensor&(at::Tensor&, const at::Tensor&, bool)>();
    constexpr c10::DispatchKeySet fallback_keyset(
        c10::DispatchKeySet(c10::DispatchKey::CompositeExplicitAutograd));
    copy_op.redispatch(fallback_keyset, out, self, /*non_blocking=*/false);
  }

  if (N == 0) return out;

  // ---- 计算 kernel 参数 (padding 到 kMaxNdim) ----
  // tensor strides in broadcast idx space
  std::vector<std::vector<int64_t>> tensor_strides;
  for (int64_t i = 0; i < m; i++) {
    tensor_strides.push_back(broadcast_strides(idx_shapes[i], idx_strides[i], idx_shape));
  }
  auto tensor_strides_2d = pad_2d(tensor_strides, kMaxNdim, kMaxNdim, int64_t(0));
  for (auto& row : tensor_strides_2d) row = pad_vec(row, kMaxNdim, int64_t(0));

  // self advanced strides/sizes
  std::vector<int64_t> self_adv_stride(kMaxNdim, 0);
  std::vector<int64_t> self_adv_size(kMaxNdim, 1);
  for (int64_t d = 0; d < m; d++) {
    self_adv_stride[d] = self.stride(d);
    self_adv_size[d] = self.size(d);
  }

  // values strides
  std::vector<int64_t> val_target_shape = idx_shape;
  val_target_shape.insert(val_target_shape.end(), suffix_shape.begin(), suffix_shape.end());
  std::vector<int64_t> val_shape(values.sizes().begin(), values.sizes().end());
  std::vector<int64_t> val_stride_vec(values.strides().begin(), values.strides().end());
  auto val_strides_full = broadcast_strides(val_shape, val_stride_vec, val_target_shape);
  auto val_adv_stride =
      pad_vec(std::vector<int64_t>(val_strides_full.begin(), val_strides_full.begin() + idx_ndim),
              kMaxNdim, int64_t(0));
  auto val_suf_stride =
      pad_vec(std::vector<int64_t>(val_strides_full.begin() + idx_ndim, val_strides_full.end()),
              kMaxNdim, int64_t(0));

  // self suffix strides
  std::vector<int64_t> self_suf_stride(kMaxNdim, 0);
  for (int64_t d = 0; d < suf_ndim; d++) {
    self_suf_stride[d] = self.stride(m + d);
  }

  // divisors
  auto idx_div = pad_vec(trailing_divisors(idx_shape), kMaxNdim, int64_t(1));
  auto suf_div = pad_vec(trailing_divisors(suffix_shape), kMaxNdim, int64_t(1));

  // index data pointers
  std::vector<void*> idx_ptrs;
  idx_ptrs.reserve(kMaxNdim);
  for (int64_t i = 0; i < m; i++) {
    idx_ptrs.push_back(idx_tensors[i].data_ptr());
  }
  void* pad_ptr = m > 0 ? idx_ptrs[0] : nullptr;
  idx_ptrs.resize(kMaxNdim, pad_ptr);

  int64_t block = heuristic_block(N);
  int64_t grid_x = (N + block - 1) / block;

  // ---- 启动 Triton kernel ----
  // 填充 padding 的 index tensor (用第一个有效 tensor 填充, kernel 不会访问)
  while (idx_tensors.size() < static_cast<size_t>(kMaxNdim)) {
    idx_tensors.push_back(idx_tensors[0]);
  }

  const TritonJITFunction& kernel = TritonJITFunction::get_instance(
      (utils::get_triton_src_path() / "unsafe_index_put_kernel.py").string(),
      "unsafe_index_put_kernel_cpp");

  c10::DeviceGuard guard(out.device());
  auto stream = backend::getCurrentStream();
  auto raw_stream = backend::getRawStream(stream);

  kernel(raw_stream,
         static_cast<unsigned int>(grid_x),  // grid_x
         1,                                    // grid_y
         1,                                    // grid_z
         4,                                    // num_warps
         4,                                    // num_stages
         // 张量 (TritonJITFunction 自动转为 data_ptr)
         out,
         values,
         idx_tensors[0], idx_tensors[1], idx_tensors[2], idx_tensors[3],
         // index 空间除数 (IDX_NDIM 个, padded)
         idx_div[0], idx_div[1], idx_div[2], idx_div[3],
         // tensor strides (M x kMaxNdim)
         tensor_strides_2d[0][0], tensor_strides_2d[0][1], tensor_strides_2d[0][2], tensor_strides_2d[0][3],
         tensor_strides_2d[1][0], tensor_strides_2d[1][1], tensor_strides_2d[1][2], tensor_strides_2d[1][3],
         tensor_strides_2d[2][0], tensor_strides_2d[2][1], tensor_strides_2d[2][2], tensor_strides_2d[2][3],
         tensor_strides_2d[3][0], tensor_strides_2d[3][1], tensor_strides_2d[3][2], tensor_strides_2d[3][3],
         // values strides
         val_adv_stride[0], val_adv_stride[1], val_adv_stride[2], val_adv_stride[3],
         // self advanced strides/sizes
         self_adv_stride[0], self_adv_stride[1], self_adv_stride[2], self_adv_stride[3],
         self_adv_size[0], self_adv_size[1], self_adv_size[2], self_adv_size[3],
         // suffix divisors
         suf_div[0], suf_div[1], suf_div[2], suf_div[3],
         // suffix strides
         self_suf_stride[0], self_suf_stride[1], self_suf_stride[2], self_suf_stride[3],
         val_suf_stride[0], val_suf_stride[1], val_suf_stride[2], val_suf_stride[3],
         // 元信息
         idx_numel, suffix_numel, N,
         // constexpr params
         static_cast<int32_t>(m),
         static_cast<int32_t>(idx_ndim),
         static_cast<int32_t>(suf_ndim),
         accumulate,
         block);

  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Operator dispatch: 快速路径处理 all-tensor, non-None, non-bool 索引
// 不支持的配置通过 aten::_index_put_impl_ redispatch (绕过 FlagGems) 回退到原生 PyTorch
at::Tensor unsafe_index_put_cpp(const at::Tensor& self,
                                const c10::List<std::optional<at::Tensor>>& indices,
                                const at::Tensor& values,
                                bool accumulate) {
  // 快速路径: no None, no bool/byte mask, dims within limit
  bool all_tensor = true;
  int64_t m = indices.size();
  for (int64_t i = 0; i < m; i++) {
    if (!indices[i].has_value()) { all_tensor = false; break; }
    auto dt = indices[i].value().dtype();
    if (dt == at::kBool || dt == at::kByte) { all_tensor = false; break; }
  }

  if (all_tensor && m > 0 && m <= kMaxNdim) {
    return unsafe_index_put(self, indices, values, accumulate);
  }

  // 回退: 通过 CompositeExplicitAutograd redispatch 到 PyTorch 原生实现
  // (绕过 FlagGems dispatch, 适用于 None 索引、bool mask、6+ 维度等情况)
  auto out = self.clone();
  static auto index_put_impl_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("aten::_index_put_impl_", "")
          .typed<at::Tensor&(at::Tensor&, const c10::List<std::optional<at::Tensor>>&,
                             const at::Tensor&, bool, bool)>();
  constexpr c10::DispatchKeySet fallback_keyset(
      c10::DispatchKeySet(c10::DispatchKey::CompositeExplicitAutograd));
  index_put_impl_op.redispatch(fallback_keyset, out, indices, values, accumulate, /*unsafe=*/true);
  return out;
}

}  // namespace flag_gems
