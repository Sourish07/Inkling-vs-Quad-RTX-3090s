// Port of inkling-sglang's paired_all_reduce.cuh: NVLink pair sums, then PCIe.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

struct Params {
  const void* input;
  void* output;
  uint32_t vectors, rank, nvlink, pcie, slot_bytes;
  uint32_t* counter;
  uint8_t* workspaces[4];
};

union alignas(16) Vector {
  uint32_t bits[4];
  __nv_bfloat16 bf16[8];
  float fp32[4];
};

template <bool BF16>
__device__ void clear_zero(Vector& value) {
#pragma unroll
  for (int i = 0; i < 4; ++i)
    if (value.bits[i] == 0) value.bits[i] = BF16 ? 0x8000u : 0x80000000u;
}

__device__ bool has_zero(const Vector& value) {
  return value.bits[0] == 0 || value.bits[1] == 0 || value.bits[2] == 0 || value.bits[3] == 0;
}

__device__ Vector load_peer(const void* address) {
  Vector value;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.bits[0]), "=r"(value.bits[1]), "=r"(value.bits[2]), "=r"(value.bits[3])
               : "l"(address));
  return value;
}

__device__ void store_peer(void* address, const Vector& value) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
               : : "r"(value.bits[0]), "r"(value.bits[1]), "r"(value.bits[2]), "r"(value.bits[3]), "l"(address));
}

__device__ void reset_slot(void* address) {
  asm volatile("st.global.v4.b32 [%0], {0, 0, 0, 0};" : : "l"(address));
}

template <bool BF16>
__device__ Vector sum(const Vector& a, const Vector& b) {
  Vector result;
  if constexpr (BF16) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      result.bf16[i] = __float2bfloat16_rn(__bfloat162float(a.bf16[i]) + __bfloat162float(b.bf16[i]));
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) result.fp32[i] = a.fp32[i] + b.fp32[i];
  }
  return result;
}

template <bool BF16>
__global__ void paired_all_reduce_kernel(const __grid_constant__ Params p) {
  const uint32_t phase = p.counter[blockIdx.x];
  const uint32_t offset = (phase & 1) * 4 * p.slot_bytes;
  const auto slot = [&](uint32_t dst, uint32_t src, uint32_t vector) {
    return p.workspaces[dst] + offset + src * p.slot_bytes + vector * 16;
  };
  const auto poll = [&](uint32_t src, uint32_t vector) {
    Vector value;
    do { value = load_peer(slot(p.rank, src, vector)); } while (has_zero(value));
    return value;
  };
  const uint32_t tid = (blockIdx.x + gridDim.x * (threadIdx.x / 32)) * 32 + threadIdx.x % 32;
  const uint32_t stride = blockDim.x * gridDim.x;
  const auto* input = static_cast<const Vector*>(p.input);
  auto* output = static_cast<Vector*>(p.output);
  for (uint32_t i = tid; i < p.vectors; i += stride) {
    Vector value = input[i];
    clear_zero<BF16>(value);
    store_peer(slot(p.nvlink, p.rank, i), value);
  }
  for (uint32_t i = tid; i < p.vectors; i += stride) {
    Vector local = input[i];
    clear_zero<BF16>(local);
    const Vector remote = poll(p.nvlink, i);
    reset_slot(slot(p.rank, p.nvlink, i));
    Vector pair = p.rank < p.nvlink ? sum<BF16>(local, remote) : sum<BF16>(remote, local);
    clear_zero<BF16>(pair);
    store_peer(slot(p.pcie, p.rank, i), pair);
    output[i] = pair;
  }
  for (uint32_t i = tid; i < p.vectors; i += stride) {
    const Vector remote = poll(p.pcie, i);
    reset_slot(slot(p.rank, p.pcie, i));
    Vector total = sum<BF16>(output[i], remote);
    clear_zero<BF16>(total);
    output[i] = total;
  }
  __syncthreads();
  if (threadIdx.x == 0) p.counter[blockIdx.x] = phase ^ 1;
}

extern "C" int paired_allocate(void** ptr, int bytes) {
  const auto status = cudaMalloc(ptr, bytes);
  return status == cudaSuccess ? cudaMemset(*ptr, 0, bytes) : status;
}

extern "C" int paired_get_handle(void* ptr, void* handle) {
  return cudaIpcGetMemHandle(static_cast<cudaIpcMemHandle_t*>(handle), ptr);
}

extern "C" int paired_open_handle(const void* handle, void** ptr) {
  cudaIpcMemHandle_t value;
  std::memcpy(&value, handle, sizeof(value));
  return cudaIpcOpenMemHandle(ptr, value, cudaIpcMemLazyEnablePeerAccess);
}

extern "C" int paired_launch(const void* input, void* output, uint8_t** workspaces,
                             uint32_t* counter, int bytes, int slot_bytes, int rank,
                             int bf16, cudaStream_t stream) {
  Params p{input, output, uint32_t(bytes / 16), uint32_t(rank), uint32_t(rank ^ 1),
           uint32_t(3 - rank), uint32_t(slot_bytes), counter, {}};
  std::memcpy(p.workspaces, workspaces, sizeof(p.workspaces));
  const int threads = p.vectors <= 16 * 128 ? 128 : 512;
  if (bf16) paired_all_reduce_kernel<true><<<16, threads, 0, stream>>>(p);
  else paired_all_reduce_kernel<false><<<16, threads, 0, stream>>>(p);
  return cudaGetLastError();
}

extern "C" const char* paired_error(int status) { return cudaGetErrorString(cudaError_t(status)); }
