// Rows of one rank's shard of an embedding table. The table may live in pinned host
// memory, in which case the rows are read over PCIe.
// A token outside [start, start + rows) belongs to another rank: its row is zero.
// Grid: (tokens); every thread moves 16 bytes per step.

typedef long long int64_t;

extern "C" __global__ void embed_rows(
    const int64_t* __restrict__ ids,   // [tokens]
    const uint4* __restrict__ table,   // [rows, 16 * vectors] bytes, device or host
    uint4* __restrict__ out,           // [tokens, 16 * vectors] bytes
    int start, int rows, int vectors) {
  const int64_t local = ids[blockIdx.x] - start;
  const bool mine = local >= 0 && local < rows;
  const uint4* source = table + local * vectors;
  uint4* target = out + (int64_t)blockIdx.x * vectors;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x)
    target[i] = mine ? source[i] : make_uint4(0u, 0u, 0u, 0u);
}
