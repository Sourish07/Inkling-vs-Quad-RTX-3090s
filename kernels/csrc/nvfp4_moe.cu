#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include "nvfp4_dispatch.cuh"

at::Tensor nvfp4_linear(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& scale, const at::Tensor& scale2) {
  TORCH_CHECK(x.is_cuda(), "Activations must be on CUDA");
  const c10::cuda::CUDAGuard guard(x.device());
  for (const auto& tensor : {x, weight, scale, scale2}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == x.device(),
                "All tensors must be on the activation device");
    TORCH_CHECK(tensor.is_contiguous(), "All tensors must be contiguous");
  }
#if INKLING_BF16
  using compute_t = nv_bfloat16;
  constexpr auto dtype = at::kBFloat16;
#else
  using compute_t = half;
  constexpr auto dtype = at::kHalf;
#endif
  TORCH_CHECK(x.scalar_type() == dtype && x.dim() == 2,
              "X must be FP16/BF16 [tokens, input_dim] matching the compiled dtype");
  TORCH_CHECK(weight.scalar_type() == at::kInt && weight.dim() == 2,
              "Weight must be prepared int32 [K / 16, N * 2]");
  const auto m = x.size(0), k = x.size(1), n = weight.size(1) / 2;
  TORCH_CHECK(n > 0 && k > 0 && n % 128 == 0 && k % 64 == 0,
              "Marlin requires positive N divisible by 128 and K divisible by 64");
  TORCH_CHECK(weight.size(0) == k / 16 && weight.size(1) == n * 2,
              "Prepared weight shape does not match the input");
  TORCH_CHECK(scale.scalar_type() == at::kFloat8_e4m3fn && scale.dim() == 2 &&
              scale.size(0) == k / 16 && scale.size(1) == n,
              "Scale must be prepared E4M3 [K / 16, N]");
  TORCH_CHECK(scale2.scalar_type() == dtype && scale2.numel() == 1,
              "Scale2 must be a prepared scalar matching the activation dtype");
  const auto int_max = std::numeric_limits<int>::max();
  TORCH_CHECK(m <= int_max - 16 && n <= int_max && k <= int_max,
              "Dimensions exceed the kernel's int32 range");
  auto output = at::empty({m, n}, x.options());
  if (m == 0) return output;

  // ep_plan.py already gathers one expert's tokens. Represent those rows as
  // a single-expert, top_k=1 MoE launch; no external routing metadata is needed.
  const int block_size = m <= 8 ? 8 : 16;
  const int64_t padded_m = (m + block_size - 1) / block_size * block_size;
  const auto int_options = x.options().dtype(at::kInt);
  auto sorted_ids = at::arange(padded_m, int_options);
  auto expert_ids = at::zeros({padded_m / block_size}, int_options);
  auto padded_tokens = at::full({1}, padded_m, int_options);
  int sms;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device()));
  auto workspace = at::zeros({int64_t(sms) * 4}, int_options);
  int64_t tmp_size = std::min(n * padded_m, int64_t(sms) * 4 * block_size * 256);
  if (block_size == 8) tmp_size *= 2;
  auto scratch = at::empty({tmp_size}, x.options().dtype(at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  sglang::device::marlin_moe::marlin_mm<compute_t, false, false>(
      x.data_ptr(), weight.data_ptr(), output.data_ptr(), scratch.data_ptr(),
      nullptr, scale.data_ptr(), scale2.data_ptr(), nullptr, nullptr, nullptr,
      nullptr, sorted_ids.data_ptr(), expert_ids.data_ptr(), padded_tokens.data_ptr(),
      nullptr, block_size, 1, false, false, m, n, k, workspace.data_ptr(),
      sglang::host::kFE2M1f, false, false, true, false, k / 16, 16,
      x.get_device(), stream, -1, -1, sms, false, true, false, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
