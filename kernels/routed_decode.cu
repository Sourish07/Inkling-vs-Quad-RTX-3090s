// Decode-time routed-expert projections: one matrix-vector product per route.
//
// Compile-time parameters:
//   K      columns of the weight matrix (the projection's input width)
//   QUANT  1: NVFP4 weights as in the checkpoint (two E2M1 codes per byte, one E4M3
//             scale per 16 columns, one FP32 scale per expert); 0: BF16 weights
//
// A thread owns 32 consecutive columns for the whole kernel, so its 32 inputs stay
// in registers and the threads of a row read that row's bytes as one contiguous
// run. Weights are never dequantized into memory: a code is looked up in a
// 16-entry shared-memory table and multiplied into an FP32 accumulator. The
// products use the exact quantized value (code x block scale x expert scale).

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;

#define THREADS 128
#define TPR (K / 32)                    // threads that share one weight row
#define RPI (THREADS / TPR)             // rows a block reads per iteration
#define LANES (TPR < 32 ? TPR : 32)     // lanes of a warp that share one row
#define WPR (TPR < 32 ? 1 : TPR / 32)   // warps that share one row
#define STRIDE (QUANT ? 6 : 2)          // pointer-table entries per expert
#define DOWN (QUANT ? 3 : 1)            // first entry of the down projection

__device__ __forceinline__ float from_bf16(uint32_t bits) {
  return __uint_as_float(bits << 16);
}

// Round to nearest even, as a cast to BF16 does.
__device__ __forceinline__ uint16_t to_bf16(float value) {
  uint32_t bits = __float_as_uint(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

__device__ __forceinline__ void load_inputs(const uint16_t* x, float* v) {
  const uint4* p = (const uint4*)x;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 q = p[j];
    v[8 * j + 0] = from_bf16(q.x & 0xffffu);
    v[8 * j + 1] = from_bf16(q.x >> 16);
    v[8 * j + 2] = from_bf16(q.y & 0xffffu);
    v[8 * j + 3] = from_bf16(q.y >> 16);
    v[8 * j + 4] = from_bf16(q.z & 0xffffu);
    v[8 * j + 5] = from_bf16(q.z >> 16);
    v[8 * j + 6] = from_bf16(q.w & 0xffffu);
    v[8 * j + 7] = from_bf16(q.w >> 16);
  }
}

#if QUANT
// Tables of the 16 E2M1 values and the 256 E4M3 values.
__device__ __forceinline__ void fill_tables(float* e2, float* e4) {
  const float magnitudes[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const int t = threadIdx.x;
  if (t < 16) e2[t] = (t & 8) ? -magnitudes[t & 7] : magnitudes[t & 7];
  for (int b = t; b < 256; b += THREADS) {
    const int exponent = (b >> 3) & 15, mantissa = b & 7;
    const float value = exponent == 0 ? ldexpf((float)mantissa, -9)
                                      : ldexpf(1.f + 0.125f * mantissa, exponent - 7);
    e4[b] = (b & 128) ? -value : value;
  }
  __syncthreads();
}

// This thread's 32 columns of one row: 16 bytes of codes and two block scales.
__device__ __forceinline__ float row_dot(const uint8_t* w_row, const uint8_t* s_row, int tc,
                                         const float* x, const float* e2, const float* e4) {
  const uint4 w = ((const uint4*)w_row)[tc];
  const uint32_t s = ((const uint16_t*)s_row)[tc];
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    a0 = fmaf(e2[(w.x >> (4 * i)) & 15u], x[i], a0);
    a1 = fmaf(e2[(w.y >> (4 * i)) & 15u], x[8 + i], a1);
    a2 = fmaf(e2[(w.z >> (4 * i)) & 15u], x[16 + i], a2);
    a3 = fmaf(e2[(w.w >> (4 * i)) & 15u], x[24 + i], a3);
  }
  return fmaf(e4[s & 255u], a0 + a1, e4[s >> 8] * (a2 + a3));
}
#else
__device__ __forceinline__ float row_dot(const uint16_t* w_row, int tc, const float* x) {
  const uint4* p = (const uint4*)w_row + 4 * tc;
  float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 q = p[j];
    a[j] = fmaf(from_bf16(q.x & 0xffffu), x[8 * j + 0], a[j]);
    a[j] = fmaf(from_bf16(q.x >> 16), x[8 * j + 1], a[j]);
    a[j] = fmaf(from_bf16(q.y & 0xffffu), x[8 * j + 2], a[j]);
    a[j] = fmaf(from_bf16(q.y >> 16), x[8 * j + 3], a[j]);
    a[j] = fmaf(from_bf16(q.z & 0xffffu), x[8 * j + 4], a[j]);
    a[j] = fmaf(from_bf16(q.z >> 16), x[8 * j + 5], a[j]);
    a[j] = fmaf(from_bf16(q.w & 0xffffu), x[8 * j + 6], a[j]);
    a[j] = fmaf(from_bf16(q.w >> 16), x[8 * j + 7], a[j]);
  }
  return (a[0] + a[1]) + (a[2] + a[3]);
}
#endif

// Sum over the lanes that share this thread's row.
__device__ __forceinline__ float lane_sum(float value) {
#pragma unroll
  for (int offset = LANES / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

// part[local row][warp of the row] <- that warp's share of W[row0 + local row] . x,
// for `rows` rows. One row is spread over TPR threads; a block reads RPI rows at once.
#if QUANT
#define ROW_SUMS(W, S)                                                              \
  for (int it = 0; it < rows / RPI; ++it) {                                         \
    const int local = it * RPI + t / TPR;                                           \
    const int64_t row = row0 + local;                                               \
    const float v = lane_sum(row_dot(W + row * (K / 2), S + row * (K / 16), tc, x, e2, e4)); \
    if (t % LANES == 0) part[local * WPR + tc / 32] = v;                             \
  }                                                                                 \
  __syncthreads();
#else
#define ROW_SUMS(W, S)                                                              \
  for (int it = 0; it < rows / RPI; ++it) {                                         \
    const int local = it * RPI + t / TPR;                                           \
    const int64_t row = row0 + local;                                               \
    const float v = lane_sum(row_dot((const uint16_t*)W + row * K, tc, x));         \
    if (t % LANES == 0) part[local * WPR + tc / 32] = v;                             \
  }                                                                                 \
  __syncthreads();
#endif

// Y[route] = silu(gate) * up, where gate and up are the even and odd rows of the
// route's expert applied to the route's token. Grid: (row tiles, routes).
// Routes whose expert has a zero pointer-table row (not on this rank) do nothing.
extern "C" __global__ void gate_up(
    const uint16_t* __restrict__ X,        // [tokens, K] BF16
    const int64_t* __restrict__ indices,   // [routes] expert of each route
    const int64_t* __restrict__ pointers,  // [experts, STRIDE] weight addresses
    uint16_t* __restrict__ Y,              // [routes, N / 2] BF16
    int top_k, int max_expert, int N, int rows) {
  __shared__ float part[THREADS * 2 * WPR];
  const int t = threadIdx.x, tc = t % TPR, route = blockIdx.y;
  const int64_t expert = indices[route];
  if (expert < 0 || expert > max_expert) return;
  const int64_t* entry = pointers + expert * STRIDE;
  const uint8_t* W = (const uint8_t*)entry[0];
  if (W == nullptr) return;
#if QUANT
  __shared__ float e2[16], e4[256];
  fill_tables(e2, e4);
  const uint8_t* S = (const uint8_t*)entry[1];
  const float scale = *(const float*)entry[2];
#else
  const float scale = 1.f;
#endif
  float x[32];
  load_inputs(X + (int64_t)(route / top_k) * K + 32 * tc, x);
  const int64_t row0 = (int64_t)blockIdx.x * rows;
  ROW_SUMS(W, S)
  if (t < rows / 2) {
    float gate = 0.f, up = 0.f;
#pragma unroll
    for (int w = 0; w < WPR; ++w) {
      gate += part[(2 * t) * WPR + w];
      up += part[(2 * t + 1) * WPR + w];
    }
    gate *= scale;
    up *= scale;
    Y[(int64_t)route * (N / 2) + row0 / 2 + t] = to_bf16(gate / (1.f + expf(-gate)) * up);
  }
}

// out[token] = sum over the token's routes of weight * (W_down . activated[route]),
// with the BF16 roundings of the unfused path: the projection, then its weighted
// value. Grid: (row tiles, tokens).
extern "C" __global__ void down_reduce(
    const uint16_t* __restrict__ A,        // [routes, K] BF16 activations
    const int64_t* __restrict__ indices,   // [routes]
    const uint16_t* __restrict__ weights,  // [routes] BF16 router weights
    const int64_t* __restrict__ pointers,  // [experts, STRIDE]
    float* __restrict__ out,               // [tokens, N] FP32
    int top_k, int max_expert, int N, int rows) {
  __shared__ float part[THREADS * WPR];
  const int t = threadIdx.x, tc = t % TPR, token = blockIdx.y;
  const int64_t row0 = (int64_t)blockIdx.x * rows;
#if QUANT
  __shared__ float e2[16], e4[256];
  fill_tables(e2, e4);
#endif
  float total = 0.f;
  for (int choice = 0; choice < top_k; ++choice) {
    const int route = token * top_k + choice;
    const int64_t expert = indices[route];
    if (expert < 0 || expert > max_expert) continue;
    const int64_t* entry = pointers + expert * STRIDE + DOWN;
    const uint8_t* W = (const uint8_t*)entry[0];
    if (W == nullptr) continue;
#if QUANT
    const uint8_t* S = (const uint8_t*)entry[1];
    const float scale = *(const float*)entry[2];
#else
    const float scale = 1.f;
#endif
    float x[32];
    load_inputs(A + (int64_t)route * K + 32 * tc, x);
    ROW_SUMS(W, S)
    if (t < rows) {
      float down = 0.f;
#pragma unroll
      for (int w = 0; w < WPR; ++w) down += part[t * WPR + w];
      down = from_bf16(to_bf16(down * scale));
      total += from_bf16(to_bf16(down * from_bf16(weights[route])));
    }
    __syncthreads();
  }
  if (t < rows) out[(int64_t)token * N + row0 + t] = total;
}
