#include <torch/extension.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>
#include <array>
#include <cmath>
#include <vector>

// Pack directly from checkpoint bytes: no unpacked weight tensor, transpose,
// advanced indexing, int64 expansion, or broadcast/reduction intermediates.
std::vector<at::Tensor> prepare_cpu(
    const at::Tensor& weight, const at::Tensor& scale,
    const at::Tensor& scale2, bool bf16) {
  TORCH_CHECK(weight.device().is_cpu() && scale.device().is_cpu() && scale2.device().is_cpu(),
              "CPU preparation requires CPU tensors");
  TORCH_CHECK(weight.scalar_type() == at::kByte && weight.dim() == 2,
              "Weight must be uint8 [N, K/2]");
  const auto n = weight.size(0), k = weight.size(1) * 2, groups = k / 16;
  TORCH_CHECK(n > 0 && k > 0 && n % 128 == 0 && k % 64 == 0, "Unsupported N/K dimensions");
  TORCH_CHECK(scale.scalar_type() == at::kFloat8_e4m3fn && scale.dim() == 2 &&
              scale.size(0) == n && scale.size(1) == groups && scale2.numel() == 1,
              "Invalid scale shapes or dtype");
  const bool pinned = weight.is_pinned();
  auto packed = at::empty({groups, n * 2}, weight.options().dtype(at::kInt).pinned_memory(pinned));
  auto processed_scale = at::empty({groups, n}, scale.options().pinned_memory(pinned));
  const auto* raw = weight.data_ptr<uint8_t>();
  auto* output = reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>());
  const auto ws0 = weight.stride(0), ws1 = weight.stride(1);
  // Keep a 64-column input panel hot in cache while traversing K. Each output
  // word gathers four bytes and places their low/high nibbles in fragment order.
  for (int64_t nb = 0; nb < n; nb += 64) {
    for (int64_t kb = 0; kb < k; kb += 16) {
      auto* out = output + (kb / 16) * n * 2 + (nb / 64) * 128;
      for (int fragment = 0; fragment < 128; ++fragment) {
        const int lane = fragment / 4;
        const int64_t column = nb + (fragment % 4) * 16 + lane / 4;
        const int64_t row = kb + 2 * (lane % 4);
        const auto a = raw[column * ws0 + (row / 2) * ws1];
        const auto b = raw[column * ws0 + ((row + 8) / 2) * ws1];
        const auto c = raw[(column + 8) * ws0 + (row / 2) * ws1];
        const auto d = raw[(column + 8) * ws0 + ((row + 8) / 2) * ws1];
        out[fragment] = uint32_t(a & 15) | (uint32_t(b & 15) << 4) |
            (uint32_t(c & 15) << 8) | (uint32_t(d & 15) << 12) |
            (uint32_t(a >> 4) << 16) | (uint32_t(b >> 4) << 20) |
            (uint32_t(c >> 4) << 24) | (uint32_t(d >> 4) << 28);
      }
    }
  }

  // The half exponent conversion used by Marlin depends only on the FP8 byte.
  static const auto scale_bits = [] {
    std::array<uint8_t, 256> result{};
    for (int i = 0; i < 256; ++i) {
      const c10::Float8_e4m3fn value(uint8_t(i), c10::Float8_e4m3fn::from_bits());
      const c10::Half scaled(float(value) * 128.0f);
      result[i] = uint8_t((uint16_t(scaled.x << 1)) >> 8);
    }
    return result;
  }();
  const auto* scales = scale.data_ptr<c10::Float8_e4m3fn>();
  auto* out_scales = processed_scale.data_ptr<c10::Float8_e4m3fn>();
  const auto ss0 = scale.stride(0), ss1 = scale.stride(1);
  constexpr int interleave[4] = {0, 2, 1, 3};
  for (int64_t nb = 0; nb < n; nb += 64) {
    for (int64_t group = 0; group < groups; ++group) {
      for (int p = 0; p < 64; ++p) {
        const int reordered = (p / 4) * 4 + interleave[p % 4];
        const int64_t column = nb + reordered / 8 + 8 * (reordered % 8);
        const auto bits = scales[column * ss0 + group * ss1].x;
        out_scales[group * n + nb + p].x = scale_bits[bits];
      }
    }
  }
  const auto dtype = bf16 ? at::kBFloat16 : at::kHalf;
  auto processed_scale2 = at::empty({1}, scale2.options().dtype(dtype).pinned_memory(pinned));
  processed_scale2.copy_(scale2.to(dtype).reshape({1}) * std::ldexp(1.0, bf16 ? 119 : 7));
  return {packed, processed_scale, processed_scale2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &prepare_cpu, "Direct CPU NVFP4 Marlin preparation");
}
