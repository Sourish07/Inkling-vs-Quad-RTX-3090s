// Minimal PyTorch/CUDA adapter for the unchanged Marlin device headers.
#pragma once
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
namespace sglang {
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
namespace host {
template <typename... Args>
[[noreturn]] inline void Panic(Args&&... args) {
  TORCH_CHECK(false, std::forward<Args>(args)...);
}
template <typename... Args>
inline void RuntimeCheck(bool condition, Args&&... args) {
  TORCH_CHECK(condition, std::forward<Args>(args)...);
}
inline void RuntimeDeviceCheck(cudaError_t status) { C10_CUDA_CHECK(status); }
}
namespace device {
template <typename T, typename U>
__host__ __device__ constexpr auto div_ceil(T a, U b) { return (a + b - 1) / b; }
}
}
