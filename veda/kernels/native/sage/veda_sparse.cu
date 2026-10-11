/*
 * Production Sol-style sparse attention for sm75 and newer architectures.
 *
 * One 64-token centroid per block feeds an input-adaptive mean + tau * std
 * threshold. Each Query CTA performs routing directly before its FP32 online
 * softmax and keeps the route in CTA-local shared memory/registers, so no full
 * global proxy or route map is materialized. Selected blocks are
 * evaluated with the production per-warp/per-block INT8 Sage QK path. Routing
 * and skipped-block correction use centroids reconstructed from those same
 * INT8 Q/K tensors and scales. Exact proxy scores stay in Sage's randomized
 * Hadamard domain, while diagonal route statistics are inverse-transformed to
 * the pre-Hadamard basis; the orthogonal transform preserves their dot
 * products but avoids estimating diagonal variance in the mixed basis. Query
 * summary, thresholding, and routing are fused into the attention CTA;
 * original V means remain isolated to the skipped-block approximation.
 * The official local neighborhood is fixed to +/- one 64-token block. Optional
 * exact-KV and dense-Query block masks carry model-independent modality policy.
 */

#include "../utils.cuh"
#include "../math.cuh"
#include "attn_utils.cuh"
#include "dispatch_utils.h"
#include "../cuda_checks.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <type_traits>

namespace {

constexpr int kBlockTokens = 64;
constexpr int kWarps = 4;
constexpr int kRouteTile = 16;
constexpr int kRouteWordBits = 32;
constexpr int kMaxKeyBlocks = 4096;
constexpr int kMaxRouteWords = kMaxKeyBlocks / kRouteWordBits;
constexpr int kSummaryTileTokens = 16;
constexpr int kMaxRouteBytes = kMaxRouteWords * sizeof(uint32_t);
constexpr int kProxyScratchBytes =
    kWarps * kSummaryTileTokens * sizeof(float);
constexpr int kCompactionScratchWords = 2 * kWarps;
constexpr int kCompactionScratchBytes =
    kCompactionScratchWords * sizeof(uint32_t);
constexpr int kSlaQueryBlockTokens = 128;

__device__ __forceinline__ bool route_convrot_negative_sign(int channel)
{
  constexpr uint32_t signs[4] = {
      0x1035997bu, 0x8087f5eeu, 0xee2e4e1au, 0x71132418u};
  return ((signs[channel >> 5] >> (channel & 31)) & 1u) == 0u;
}

// The exact QK path remains in the randomized Hadamard basis.  Routing uses
// the inverse-transformed block centroids so its diagonal variance model is
// evaluated in the pre-Hadamard basis.  The first five butterfly stages stay
// within a warp; D64/D128 need only one/two cross-warp shared-memory stages.
template <int HeadDim>
__device__ __forceinline__ float inverse_route_hadamard(
    float value,
    float *__restrict__ scratch)
{
  const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = linear_thread & (WARP_SIZE - 1);
#pragma unroll
  for (int bit = 1; bit < WARP_SIZE; bit <<= 1)
  {
    const float other = __shfl_xor_sync(0xffffffffu, value, bit);
    value = (lane & bit) ? other - value : value + other;
  }
#pragma unroll
  for (int bit = WARP_SIZE; bit < HeadDim; bit <<= 1)
  {
    if (linear_thread < HeadDim)
      scratch[linear_thread] = value;
    __syncthreads();
    float other = 0.0f;
    if (linear_thread < HeadDim)
      other = scratch[linear_thread ^ bit];
    __syncthreads();
    if (linear_thread < HeadDim)
      value = (linear_thread & bit) ? other - value : value + other;
  }
  constexpr float scale = HeadDim == 64
      ? 0.125f
      : 0.08838834764831845f;
  value *= scale;
  if (linear_thread < HeadDim && route_convrot_negative_sign(linear_thread))
    value = -value;
  return value;
}

template <int HeadDim>
struct AttentionGeometry
{
  static_assert(HeadDim == 64 || HeadDim == 128);
  static constexpr int kHalfPacks = HeadDim / 8;
  static constexpr int kInt8Packs = HeadDim / 16;
  static constexpr int kTilePacks = kBlockTokens * kHalfPacks;
  static constexpr int kInt8TilePacks = kBlockTokens * kInt8Packs;
  static constexpr int kTileBytes = kBlockTokens * HeadDim * sizeof(half);
  static constexpr int kInt8TileBytes = kBlockTokens * HeadDim * sizeof(int8_t);
  static constexpr int kSummaryTileBytes =
      kSummaryTileTokens * HeadDim * sizeof(half);
  static constexpr int kAttentionSharedBytes = 2 * kTileBytes;
  static constexpr int kRouteStorageOffset =
      kTileBytes + 2 * kSummaryTileBytes;
  static constexpr int kSelectedStorageOffset =
      kRouteStorageOffset + kMaxRouteBytes;
  static constexpr int kSelectedCapacity =
      (kAttentionSharedBytes - kSelectedStorageOffset - sizeof(int) -
       kCompactionScratchBytes) /
      sizeof(uint16_t);
  static constexpr int kCompactionScratchOffset =
      kSelectedStorageOffset + sizeof(int) +
      kSelectedCapacity * sizeof(uint16_t);
  static constexpr int kValueTiles = HeadDim / 16;
  static constexpr SwizzleMode kInt8Swizzle =
      HeadDim == 64 ? SwizzleMode::k64B : SwizzleMode::k128B;
  static_assert(
      kRouteStorageOffset + kMaxRouteBytes + kProxyScratchBytes <=
          kAttentionSharedBytes,
      "fused routing metadata must fit beside the 16-block summaries");
  static_assert(kSelectedCapacity >= 1024,
                "sparse route compaction must cover production sequences");
  static_assert(
      kCompactionScratchOffset + kCompactionScratchBytes <=
          kAttentionSharedBytes,
      "sparse route compaction scratch must fit in shared memory");
};

static_assert(AttentionGeometry<128>::kAttentionSharedBytes <= 64 * 1024);

template <int HeadDim, bool SparseValuePipeline>
struct AttentionStorage
{
  using G = AttentionGeometry<HeadDim>;
  // A sparse SM80+ V pipeline needs one additional INT8 V tile.  Moving route
  // metadata behind that tile keeps the exact phase's two V stages disjoint
  // from the stable compact list.  The default SM75/storage layout is
  // unchanged and remains exactly two 32 KiB CTAs per 64 KiB SM at D128.
  static constexpr int kRouteStorageOffset = SparseValuePipeline
      ? G::kAttentionSharedBytes
      : G::kRouteStorageOffset;
  static constexpr int kAttentionSharedBytes = G::kAttentionSharedBytes +
      (SparseValuePipeline ? G::kInt8TileBytes : 0);
  static constexpr int kSelectedStorageOffset =
      kRouteStorageOffset + kMaxRouteBytes;
  static constexpr int kSelectedCapacity =
      (kAttentionSharedBytes - kSelectedStorageOffset - sizeof(int) -
       kCompactionScratchBytes) /
      sizeof(uint16_t);
  static constexpr int kCompactionScratchOffset =
      kSelectedStorageOffset + sizeof(int) +
      kSelectedCapacity * sizeof(uint16_t);
  static_assert(
      !SparseValuePipeline ||
          (G::kRouteStorageOffset >= 3 * G::kInt8TileBytes &&
           G::kRouteStorageOffset + 2 * G::kSummaryTileBytes <=
               4 * G::kInt8TileBytes),
      "pipelined summaries must fit in the later reused exact-V stage");
  static_assert(
      !SparseValuePipeline ||
          4 * G::kInt8TileBytes <= G::kAttentionSharedBytes,
      "two exact-V stages must end before pipelined route metadata");
  static_assert(kSelectedCapacity >= 1024,
                "attention storage must cover production sparse routes");
  static_assert(
      kCompactionScratchOffset + kCompactionScratchBytes <=
          kAttentionSharedBytes,
      "attention storage exceeds dynamic shared memory");
};

template <typename T>
__device__ __forceinline__ float scalar_to_float(T value);

template <>
__device__ __forceinline__ float scalar_to_float<half>(half value)
{
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<nv_bfloat16>(nv_bfloat16 value)
{
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ b128_t pack_to_half(const T *source);

template <>
__device__ __forceinline__ b128_t pack_to_half<half>(const half *source)
{
  return *reinterpret_cast<const b128_t *>(source);
}

template <>
__device__ __forceinline__ b128_t pack_to_half<nv_bfloat16>(const nv_bfloat16 *source)
{
  return bf16_pack_to_half(source);
}

template <int HeadDim, typename T, bool NormalizeValue, bool MappedValue = false>
__global__ void kv_block_summary_kernel(
    const int8_t *__restrict__ key_int8,
    const float *__restrict__ key_scale,
    const T *__restrict__ value,
    const float *__restrict__ value_scale,
    const int *__restrict__ value_source_indices,
    half *__restrict__ key_summary,
    half *__restrict__ key_score_summary,
    half *__restrict__ value_mean,
    int batch_size,
    int num_heads,
    int sequence_length,
    int padded_blocks,
    int residual_subblocks,
    int route_original_basis,
    int padded_residual_summaries,
    int64_t stride_batch_k_int8,
    int64_t stride_head_k_int8,
    int64_t stride_sequence_k_int8,
    int64_t stride_batch_v,
    int64_t stride_head_v,
    int64_t stride_sequence_v)
{
  const int block_index = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int dimension = threadIdx.x;
  __shared__ float inverse_scratch[HeadDim];

  const int token_start = block_index * kBlockTokens;
  const int token_count = token_start < sequence_length
      ? min(kBlockTokens, sequence_length - token_start)
      : 0;
  float value_sum[2] = {0.0f, 0.0f};
  int quantized_key_sum[2] = {0, 0};
  float route_key_mean = 0.0f;
  const int8_t *head_key_int8 = key_int8 +
      batch * stride_batch_k_int8 + head * stride_head_k_int8;
  const T *head_value = value + batch * stride_batch_v + head * stride_head_v;
  for (int token = 0; token < token_count; ++token)
  {
    const int logical_token = token_start + token;
    const int value_token = MappedValue
        ? value_source_indices[logical_token]
        : logical_token;
    const int residual_index = token / (kBlockTokens / residual_subblocks);
    quantized_key_sum[residual_index] += static_cast<int>(
        head_key_int8[logical_token * stride_sequence_k_int8 + dimension]);
    value_sum[residual_index] += scalar_to_float(
        head_value[value_token * stride_sequence_v + dimension]);
  }
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) * padded_blocks + block_index) *
          HeadDim +
      dimension;
  if (token_count)
  {
    const int num_key_blocks = (sequence_length + kBlockTokens - 1) / kBlockTokens;
    const float dequant_scale = key_scale[
        (static_cast<int64_t>(batch) * num_heads + head) * num_key_blocks +
        block_index];
    const int total_quantized_key_sum = quantized_key_sum[0] +
        (residual_subblocks == 2 ? quantized_key_sum[1] : 0);
    route_key_mean = static_cast<float>(total_quantized_key_sum) *
        dequant_scale / static_cast<float>(token_count);
    const int residual_tokens = kBlockTokens / residual_subblocks;
#pragma unroll
    for (int residual_index = 0; residual_index < 2; ++residual_index)
    {
      if (residual_index >= residual_subblocks)
        break;
      const int residual_start = residual_index * residual_tokens;
      const int residual_count = max(0, min(residual_tokens, token_count - residual_start));
      const int64_t residual_output_index =
          ((static_cast<int64_t>(batch) * num_heads + head) *
               padded_residual_summaries +
           block_index * residual_subblocks + residual_index) *
              HeadDim +
          dimension;
      if (residual_count > 0)
      {
        const float residual_reciprocal = 1.0f / residual_count;
        if (key_score_summary != key_summary || residual_subblocks != 1)
        {
          key_score_summary[residual_output_index] = __float2half_rn(
              static_cast<float>(quantized_key_sum[residual_index]) *
              dequant_scale * residual_reciprocal);
        }
        float mean = value_sum[residual_index] * residual_reciprocal;
        if constexpr (NormalizeValue)
        {
          const float channel_scale = value_scale[
              (static_cast<int64_t>(batch) * num_heads + head) * HeadDim +
              dimension];
          mean /= channel_scale;
        }
        value_mean[residual_output_index] = __float2half_rn(mean);
      }
      else
      {
        if (key_score_summary != key_summary || residual_subblocks != 1)
          key_score_summary[residual_output_index] = __float2half_rn(0.0f);
        value_mean[residual_output_index] = __float2half_rn(0.0f);
      }
    }
  }
  else
  {
    if (block_index * residual_subblocks < padded_residual_summaries)
    {
#pragma unroll
      for (int residual_index = 0; residual_index < 2; ++residual_index)
      {
        if (residual_index >= residual_subblocks)
          break;
        const int64_t residual_output_index =
            ((static_cast<int64_t>(batch) * num_heads + head) *
                 padded_residual_summaries +
             block_index * residual_subblocks + residual_index) *
                HeadDim +
            dimension;
        if (key_score_summary != key_summary || residual_subblocks != 1)
          key_score_summary[residual_output_index] = __float2half_rn(0.0f);
        value_mean[residual_output_index] = __float2half_rn(0.0f);
      }
    }
  }
  if (route_original_basis)
    route_key_mean = inverse_route_hadamard<HeadDim>(
        route_key_mean, inverse_scratch);
  key_summary[output_index] = __float2half_rn(route_key_mean);
}

template <int HeadDim>
__global__ void key_summary_stats_kernel(
    const half *__restrict__ key_summary,
    float *__restrict__ key_summary_mean,
    float *__restrict__ key_summary_variance,
    int num_key_blocks,
    int padded_key_blocks)
{
  const int packed_head = blockIdx.x;
  const int dimension = threadIdx.x;
  if (dimension >= HeadDim)
    return;

  const half *head_summary =
      key_summary + static_cast<int64_t>(packed_head) * padded_key_blocks * HeadDim;
  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int key_block = 0; key_block < num_key_blocks; ++key_block)
  {
    const float value = __half2float(
        head_summary[static_cast<int64_t>(key_block) * HeadDim + dimension]);
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  const float reciprocal = 1.0f / static_cast<float>(num_key_blocks);
  const float mean = sum * reciprocal;
  const int64_t output_index =
      static_cast<int64_t>(packed_head) * HeadDim + dimension;
  key_summary_mean[output_index] = mean;
  key_summary_variance[output_index] =
      fmaxf(square_sum * reciprocal - mean * mean, 0.0f);
}

template <int HeadDim>
__global__ void sla_query_summary_kernel(
    const int8_t *__restrict__ query_int8,
    const float *__restrict__ query_scale,
    half *__restrict__ query_summary,
    int query_length,
    int num_heads,
    int num_query_blocks_64,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence)
{
  const int query_block = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int dimension = threadIdx.x;
  const int token_start = query_block * kSlaQueryBlockTokens;
  const int token_count = min(kSlaQueryBlockTokens, query_length - token_start);
  const int8_t *head_query = query_int8 +
      batch * stride_batch + head * stride_head;
  const float *head_scale = query_scale +
      (static_cast<int64_t>(batch) * num_heads + head) *
          num_query_blocks_64 * kWarps;
  float sum = 0.0f;
  for (int token = 0; token < token_count; ++token)
  {
    const int global_token = token_start + token;
    const float scale = head_scale[
        (global_token / kBlockTokens) * kWarps +
        (global_token % kBlockTokens) / (kBlockTokens / kWarps)];
    sum = fmaf(
        static_cast<float>(
            head_query[static_cast<int64_t>(global_token) * stride_sequence +
                       dimension]),
        scale,
        sum);
  }
  const int num_query_blocks_128 =
      div_ceil(query_length, kSlaQueryBlockTokens);
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) *
           num_query_blocks_128 +
       query_block) *
          HeadDim +
      dimension;
  query_summary[output_index] = __float2half_rn(
      sum / static_cast<float>(token_count));
}

template <int HeadDim>
__global__ void sla_key_summary_kernel(
    const int8_t *__restrict__ key_int8,
    const float *__restrict__ key_scale,
    half *__restrict__ key_summary,
    int key_length,
    int num_heads,
    int num_key_blocks,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence)
{
  const int key_block = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int dimension = threadIdx.x;
  const int token_start = key_block * kBlockTokens;
  const int token_count = min(kBlockTokens, key_length - token_start);
  const int8_t *head_key = key_int8 + batch * stride_batch + head * stride_head;
  int quantized_sum = 0;
  for (int token = 0; token < token_count; ++token)
  {
    quantized_sum += static_cast<int>(
        head_key[static_cast<int64_t>(token_start + token) * stride_sequence +
                 dimension]);
  }
  const float scale = key_scale[
      (static_cast<int64_t>(batch) * num_heads + head) * num_key_blocks +
      key_block];
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) * num_key_blocks +
       key_block) *
          HeadDim +
      dimension;
  key_summary[output_index] = __float2half_rn(
      static_cast<float>(quantized_sum) * scale /
      static_cast<float>(token_count));
}

__global__ void sla_topk_route_kernel(
    const int32_t *__restrict__ topk_indices,
    uint32_t *__restrict__ route_words,
    int64_t index_count,
    int topk,
    int route_word_count,
    int num_key_blocks)
{
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= index_count)
    return;
  const int key_block = topk_indices[index];
  if (key_block < 0 || key_block >= num_key_blocks)
    return;
  const int64_t route_row = index / topk;
  atomicOr(
      route_words + route_row * route_word_count +
          key_block / kRouteWordBits,
      1U << (key_block % kRouteWordBits));
}

__global__ void sla_exact_route_kernel(
    const uint8_t *__restrict__ exact_kv_blocks,
    uint32_t *__restrict__ route_words,
    int64_t route_rows,
    int route_word_count,
    int num_key_blocks)
{
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_words = route_rows * route_word_count;
  if (index >= total_words)
    return;
  const int word_index = index % route_word_count;
  uint32_t exact_word = 0;
#pragma unroll
  for (int bit = 0; bit < kRouteWordBits; ++bit)
  {
    const int key_block = word_index * kRouteWordBits + bit;
    if (key_block < num_key_blocks && exact_kv_blocks[key_block])
      exact_word |= 1U << bit;
  }
  route_words[index] |= exact_word;
}

__device__ __forceinline__ void block_reduce_pair(
    float &first,
    float &second,
    float *__restrict__ scratch)
{
  const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = linear_thread % WARP_SIZE;
  const int warp = linear_thread / WARP_SIZE;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
  {
    first += __shfl_down_sync(0xffffffff, first, offset);
    second += __shfl_down_sync(0xffffffff, second, offset);
  }
  if (lane == 0)
  {
    scratch[warp] = first;
    scratch[kWarps + warp] = second;
  }
  __syncthreads();
  if (warp == 0)
  {
    first = lane < kWarps ? scratch[lane] : 0.0f;
    second = lane < kWarps ? scratch[kWarps + lane] : 0.0f;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
      first += __shfl_down_sync(0xffffffff, first, offset);
      second += __shfl_down_sync(0xffffffff, second, offset);
    }
    if (lane == 0)
    {
      scratch[0] = first;
      scratch[1] = second;
    }
  }
  __syncthreads();
  first = scratch[0];
  second = scratch[1];
}

template <int HeadDim, int Rows, typename T>
__device__ __forceinline__ void load_half_tile(
    const T *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  static_assert(Rows > 0 && Rows <= kBlockTokens && Rows % 16 == 0);
  constexpr int tile_packs = Rows * G::kHalfPacks;
  for (int line = linear_thread; line < tile_packs; line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kHalfPacks;
    const int column = line % G::kHalfPacks;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    if (row_start + row < row_limit)
    {
      destination.base[offset] =
          pack_to_half(source + static_cast<int64_t>(row_start + row) * stride_sequence + column * 8);
    }
    else
    {
      destination.base[offset] = make_uint4(0, 0, 0, 0);
    }
  }
}

template <int HeadDim, int Rows, typename T>
__device__ __forceinline__ void load_half_tile_mapped(
    const T *__restrict__ source,
    int64_t stride_sequence,
    const int *__restrict__ source_indices,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  static_assert(Rows > 0 && Rows <= kBlockTokens && Rows % 16 == 0);
  constexpr int tile_packs = Rows * G::kHalfPacks;
  for (int line = linear_thread; line < tile_packs;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kHalfPacks;
    const int column = line % G::kHalfPacks;
    const int logical_row = row_start + row;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    if (logical_row < row_limit)
    {
      const int physical_row = source_indices[logical_row];
      destination.base[offset] = pack_to_half(
          source + static_cast<int64_t>(physical_row) * stride_sequence +
          column * 8);
    }
    else
    {
      destination.base[offset] = make_uint4(0, 0, 0, 0);
    }
  }
}

template <int HeadDim, int Rows>
__device__ __forceinline__ void load_half_tile_async(
    const half *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  static_assert(Rows > 0 && Rows <= kBlockTokens && Rows % 16 == 0);
  constexpr int tile_packs = Rows * G::kHalfPacks;
  for (int line = linear_thread; line < tile_packs;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kHalfPacks;
    const int column = line % G::kHalfPacks;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    const half *source_line = source +
        static_cast<int64_t>(row_start + row) * stride_sequence + column * 8;
    destination.template load_128b_async<
        cp_async::SharedMemFillMode::kFillZero>(
        offset, source_line, row_start + row < row_limit);
  }
}

template <int HeadDim>
__device__ __forceinline__ void load_int8_tile(
    const int8_t *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < G::kInt8TilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kInt8Packs;
    const int column = line % G::kInt8Packs;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    if (row_start + row < row_limit)
    {
      destination.base[offset] = *reinterpret_cast<const b128_t *>(
          source + static_cast<int64_t>(row_start + row) * stride_sequence +
          column * 16);
    }
    else
    {
      destination.base[offset] = make_uint4(0, 0, 0, 0);
    }
  }
}

template <int HeadDim>
__device__ __forceinline__ void load_int8_tile_async(
    const int8_t *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < G::kInt8TilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kInt8Packs;
    const int column = line % G::kInt8Packs;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    const int8_t *source_line = source +
        static_cast<int64_t>(row_start + row) * stride_sequence +
        column * 16;
    destination.template load_128b_async<
        cp_async::SharedMemFillMode::kFillZero>(
        offset, source_line, row_start + row < row_limit);
  }
}

template <int HeadDim>
__device__ __forceinline__ void dequantize_int8_tile(
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &source,
    const float *__restrict__ scale,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < G::kTilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kHalfPacks;
    const int column = line % G::kHalfPacks;
    const int global_row = row_start + row;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    b128_t packed = make_uint4(0, 0, 0, 0);
    if (global_row < row_limit)
    {
      const float dequant_scale = scale[
          (global_row / kBlockTokens) * kWarps +
          (global_row % kBlockTokens) / (kBlockTokens / kWarps)];
      half *packed_half = reinterpret_cast<half *>(&packed);
#pragma unroll
      for (int element = 0; element < 8; ++element)
      {
        const int dimension = column * 8 + element;
        const uint32_t source_offset = source.get_permuted_offset(
            row, dimension / 16);
        const int8_t quantized = reinterpret_cast<const int8_t *>(
            source.base + source_offset)[dimension % 16];
        packed_half[element] = __float2half_rn(
            static_cast<float>(quantized) * dequant_scale);
      }
    }
    destination.base[offset] = packed;
  }
}

template <int HeadDim, int KeyTiles>
__device__ __forceinline__ void compute_fp16_qk(
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &query,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &key,
    float score[1][KeyTiles][8])
{
  using G = AttentionGeometry<HeadDim>;
  static_assert(KeyTiles == 1 || KeyTiles == 2 || KeyTiles == 4);
  uint32_t query_offset = query.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x % 16, threadIdx.x / 16);
  uint32_t key_offset = key.get_permuted_offset(
      threadIdx.x % 8 + (threadIdx.x / 16) * 8,
      (threadIdx.x / 8) % 2);

#pragma unroll
  for (int inner = 0; inner < HeadDim / 16; ++inner)
  {
    uint32_t query_fragment[4];
    query.ldmatrix_m8n8x4(query_offset, query_fragment);
    query_offset = query.advance_offset_by_row<16>(query_offset);
    query_offset = query.advance_offset_by_column<2>(
        query_offset - 16 * G::kHalfPacks, inner);

#pragma unroll
    for (int key_tile = 0; key_tile < KeyTiles; ++key_tile)
    {
      uint32_t key_fragment[4];
      key.ldmatrix_m8n8x4(key_offset, key_fragment);
      key_offset = key.advance_offset_by_row<16>(key_offset);
      if (inner == 0)
      {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<mma::MMAMode::kInit>(
            score[0][key_tile], query_fragment, key_fragment);
      }
      else
      {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<mma::MMAMode::kInplaceUpdate>(
            score[0][key_tile], query_fragment, key_fragment);
      }
    }
    key_offset = key.advance_offset_by_column<2>(
        key_offset - KeyTiles * 16 * G::kHalfPacks, inner);
  }
}

template <int HeadDim>
__device__ __forceinline__ void compute_int8_qk(
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &query,
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &key,
    int32_t score[1][4][8])
{
  using G = AttentionGeometry<HeadDim>;
  uint32_t query_offset = query.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x % 16, threadIdx.x / 16);
  uint32_t key_offset = key.get_permuted_offset(
      threadIdx.x % 8 + (threadIdx.x / 16) * 8,
      (threadIdx.x / 8) % 2);
  compute_int_qk<4, 1, 1, 4, HeadDim / 32,
                 G::kInt8Swizzle, G::kInt8Packs, DataType::kInt8>(
      query, key, score, query_offset, key_offset);
}

#include "sparse/route_compaction.cuh"

template <int HeadDim>
__device__ __forceinline__ void load_quantized_value_tile(
    const int8_t *__restrict__ value,
    int padded_sequence_length,
    int key_block,
    smem_t<SwizzleMode::k64B, 4> shared_value)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  constexpr int lines = HeadDim * kBlockTokens / 16;
#pragma unroll
  for (int line = linear_thread; line < lines; line += kWarps * WARP_SIZE)
  {
    const int channel = line / 4;
    const int sequence_pack = line % 4;
    const uint32_t destination = shared_value.get_permuted_offset(
        channel, sequence_pack);
    const int8_t *source = value +
        static_cast<int64_t>(channel) * padded_sequence_length +
        key_block * kBlockTokens + sequence_pack * 16;
    shared_value.base[destination] = *reinterpret_cast<const b128_t *>(source);
  }
}

template <int HeadDim>
__device__ __forceinline__ void load_quantized_value_tile_async(
    const int8_t *__restrict__ value,
    int padded_sequence_length,
    int key_block,
    smem_t<SwizzleMode::k64B, 4> shared_value)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  constexpr int lines = HeadDim * kBlockTokens / 16;
#pragma unroll
  for (int line = linear_thread; line < lines; line += kWarps * WARP_SIZE)
  {
    const int channel = line / 4;
    const int sequence_pack = line % 4;
    const uint32_t destination = shared_value.get_permuted_offset(
        channel, sequence_pack);
    const int8_t *source = value +
        static_cast<int64_t>(channel) * padded_sequence_length +
        key_block * kBlockTokens + sequence_pack * 16;
    shared_value.load_128b_async(destination, source);
  }
}

template <int HeadDim, typename T, bool UseW8A8, bool ForceDense,
          bool IsCausal, bool Varlen, int ResidualSubblocks, int KeyStages,
          bool ExternalRoute = false, bool SparseValuePipeline = false,
          bool MappedValue = false>
__global__ void sparse_attention_kernel(
    const int8_t *__restrict__ query_int8,
    const int8_t *__restrict__ key_int8,
    const T *__restrict__ value,
    const int8_t *__restrict__ value_int8,
    const float *__restrict__ value_scale,
    T *__restrict__ output,
    const float *__restrict__ query_scale,
    const float *__restrict__ key_scale,
    const half *__restrict__ key_score_summary,
    const half *__restrict__ value_mean,
    const float *__restrict__ key_summary_mean,
    const float *__restrict__ key_summary_variance,
    const uint8_t *__restrict__ sparse_query_blocks,
    const uint8_t *__restrict__ exact_kv_blocks,
    const uint32_t *__restrict__ external_route_words,
    unsigned long long *__restrict__ selected_count,
    const int32_t *__restrict__ cu_seqlens_q,
    const int32_t *__restrict__ cu_seqlens_k,
    const int32_t *__restrict__ value_offsets,
    const int32_t *__restrict__ value_source_indices,
    int query_length,
    int key_length,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int padded_residual_summaries,
    int64_t stride_batch_q_int8,
    int64_t stride_head_q_int8,
    int64_t stride_sequence_q_int8,
    int64_t stride_batch_k_int8,
    int64_t stride_head_k_int8,
    int64_t stride_sequence_k_int8,
    int64_t stride_batch_v,
    int64_t stride_head_v,
    int64_t stride_sequence_v,
    int padded_value_length,
    int total_value_length,
    int64_t stride_batch_o,
    int64_t stride_head_o,
    int64_t stride_sequence_o,
    float threshold_sigma,
    float softmax_scale,
    int route_original_basis)
{
  using G = AttentionGeometry<HeadDim>;
  using S = AttentionStorage<HeadDim, SparseValuePipeline>;
  static_assert(
      ResidualSubblocks == 1 || ResidualSubblocks == 2,
      "Sol residual geometry must be 1x64 or 2x32");
  static_assert(
      KeyStages == 1 || KeyStages == 2,
      "exact attention stages must cover 64 or 128 K tokens");
  static_assert(!ForceDense || KeyStages == 1);
  static_assert(!ExternalRoute || (!ForceDense && !Varlen && !IsCausal));
  static_assert(
      !MappedValue || (!UseW8A8 && !ForceDense && !Varlen && !IsCausal),
      "mapped FP16/BF16 V is scoped to non-causal split Sol");
  static_assert(
      !SparseValuePipeline ||
          (UseW8A8 && !ForceDense && !Varlen && !IsCausal),
      "the extra V stage is scoped to non-causal W8A8 sparse routes");
  static_assert(!IsCausal || ForceDense,
                "causal masking is supported only by dense W8A8");
  static_assert(
      S::kAttentionSharedBytes <= 64 * 1024,
      "sparse attention exceeds the configured shared-memory limit");
  extern __shared__ int8_t shared_bytes[];
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_correction_query(shared_bytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_summary_key(
      shared_bytes + G::kTileBytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_summary_value(
      shared_bytes + G::kTileBytes + G::kSummaryTileBytes);
  // Routing finishes before exact PV starts, so the alternate summary pair
  // intentionally aliases the alternate INT8 V tile across those two phases.
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_summary_key_next(
      shared_bytes + G::kRouteStorageOffset);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_summary_value_next(
      shared_bytes + G::kRouteStorageOffset + G::kSummaryTileBytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_output(shared_bytes);
  smem_t<G::kInt8Swizzle, G::kInt8Packs> shared_query_int8(shared_bytes);
  smem_t<G::kInt8Swizzle, G::kInt8Packs> shared_initial_query_int8(
      shared_bytes + G::kTileBytes);
  smem_t<G::kInt8Swizzle, G::kInt8Packs> shared_key_int8(
      shared_bytes + G::kInt8TileBytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_selected_value(
      shared_bytes + 2 * G::kInt8TileBytes);
  smem_t<SwizzleMode::k64B, 4> shared_selected_value_int8(
      shared_bytes + 2 * G::kInt8TileBytes);
  // Dense exact attention no longer needs the routing/selection storage once
  // it enters the K/V loop.  On sm80+ reuse that final INT8-tile region as a
  // second V stage so cp.async can overlap the next V load with the current
  // probability x V MMA.  Sparse queries retain the compact selected-block
  // list in this region; keeping their 32 KiB footprint preserves the third
  // resident CTA on GA10x instead of trading occupancy for a 40 KiB buffer.
  smem_t<SwizzleMode::k64B, 4> shared_selected_value_int8_next(
      shared_bytes + 3 * G::kInt8TileBytes);
  uint32_t *shared_route = reinterpret_cast<uint32_t *>(
      shared_bytes + S::kRouteStorageOffset);
  int *shared_selected_count = reinterpret_cast<int *>(
      shared_bytes + S::kSelectedStorageOffset);
  uint16_t *shared_selected_blocks = reinterpret_cast<uint16_t *>(
      shared_bytes + S::kSelectedStorageOffset + sizeof(int));
  uint32_t *shared_compaction_scratch = reinterpret_cast<uint32_t *>(
      shared_bytes + S::kCompactionScratchOffset);

  const int query_block = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  int query_start = 0;
  int key_start = 0;
  if constexpr (Varlen)
  {
    query_start = cu_seqlens_q[batch];
    key_start = cu_seqlens_k[batch];
    query_length = cu_seqlens_q[batch + 1] - query_start;
    key_length = cu_seqlens_k[batch + 1] - key_start;
    if (query_block * kBlockTokens >= query_length)
      return;
  }
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const int full_key_blocks = Varlen
      ? (key_length + kBlockTokens - 1) / kBlockTokens
      : num_key_blocks;
  const int active_key_blocks = IsCausal
      ? min(full_key_blocks, query_block + 1)
      : full_key_blocks;
  const bool sparse_query = ForceDense
      ? false
      : sparse_query_blocks[query_block] != 0;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int8_t *query_int8_head_ptr = query_int8 +
      (Varlen ? static_cast<int64_t>(query_start) * stride_sequence_q_int8
              : batch * stride_batch_q_int8) +
      query_head * stride_head_q_int8;
  const int8_t *key_int8_head_ptr = key_int8 +
      (Varlen ? static_cast<int64_t>(key_start) * stride_sequence_k_int8
              : batch * stride_batch_k_int8) +
      kv_head * stride_head_k_int8;
  const T *value_head_ptr =
      value + batch * stride_batch_v + kv_head * stride_head_v;
  const int8_t *value_int8_head_ptr = UseW8A8
      ? value_int8 + (Varlen
          ? static_cast<int64_t>(kv_head) * HeadDim * total_value_length +
              value_offsets[batch]
          : static_cast<int64_t>(batch * num_kv_heads + kv_head) *
              HeadDim * padded_value_length)
      : nullptr;
  const float *value_scale_head = UseW8A8
      ? value_scale +
          static_cast<int64_t>(batch * num_kv_heads + kv_head) * HeadDim
      : nullptr;
  T *output_head_ptr = output +
      (Varlen ? static_cast<int64_t>(query_start) * stride_sequence_o
              : batch * stride_batch_o) +
      query_head * stride_head_o;
  const float *query_scale_head = query_scale +
      (static_cast<int64_t>(batch) * num_query_heads + query_head) *
          num_query_blocks * kWarps;
  const float *key_scale_head = key_scale +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * num_key_blocks;

  const float q_dequant_scale =
      query_scale_head[query_block * kWarps + threadIdx.y];
  const float scale_log2 = softmax_scale * math::log2e;
  const uint32_t value_mma_offset = shared_summary_value.get_permuted_offset(
      threadIdx.x % 16, threadIdx.x / 16);
  float output_fragment[1][G::kValueTiles][8];
  float row_max[1][2];
  float denominator[1][2];
#pragma unroll
  for (int value_tile = 0; value_tile < G::kValueTiles; ++value_tile)
  {
#pragma unroll
    for (int element = 0; element < 8; ++element)
      output_fragment[0][value_tile][element] = 0.0f;
  }
  row_max[0][0] = -5000000.0f;
  row_max[0][1] = -5000000.0f;
  denominator[0][0] = 1.0f;
  denominator[0][1] = 1.0f;
  if constexpr (!ForceDense)
  {
  if constexpr (ExternalRoute)
  {
    const int route_word_count =
        (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
    const int sla_query_blocks =
        (num_query_blocks + 1) / 2;
    const uint32_t *route_head = external_route_words +
        ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
             sla_query_blocks +
         query_block / 2) *
            route_word_count;
    for (int word = linear_thread; word < route_word_count;
         word += kWarps * WARP_SIZE)
      shared_route[word] = route_head[word];
    __syncthreads();
  }
  else
  {
  // Route from the same INT8 Q and per-16-token scales consumed by exact Sage.
  // Keeping this tile in shared memory also avoids another global Q read when
  // constructing the correction operand below.
  load_int8_tile_async<HeadDim>(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_initial_query_int8);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  const int query_token_start = query_block * kBlockTokens;
  const int query_token_count = min(kBlockTokens, query_length - query_token_start);
  const int dimension = linear_thread;
  float query_sum = 0.0f;
  if (dimension < HeadDim)
  {
#pragma unroll
    for (int warp_group = 0; warp_group < kWarps; ++warp_group)
    {
      int quantized_sum = 0;
#pragma unroll
      for (int row = 0; row < kBlockTokens / kWarps; ++row)
      {
        const int token = warp_group * (kBlockTokens / kWarps) + row;
        if (token < query_token_count)
        {
          const uint32_t source_offset = shared_initial_query_int8.get_permuted_offset(
              token, dimension / 16);
          quantized_sum += static_cast<int>(reinterpret_cast<const int8_t *>(
              shared_initial_query_int8.base + source_offset)[dimension % 16]);
        }
      }
      const float dequant_scale = query_scale_head[
          query_block * kWarps + warp_group];
      query_sum = fmaf(static_cast<float>(quantized_sum), dequant_scale, query_sum);
    }
  }
  float query_mean = query_sum / static_cast<float>(query_token_count);

  float *reduction_scratch = reinterpret_cast<float *>(
      shared_bytes + S::kRouteStorageOffset);
  if (route_original_basis)
    query_mean = inverse_route_hadamard<HeadDim>(
        query_mean, reduction_scratch);
  const float *key_mean = key_summary_mean +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * HeadDim;
  const float *key_variance = key_summary_variance +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * HeadDim;
  float projected_mean = 0.0f;
  float projected_variance = 0.0f;
  if (dimension < HeadDim)
  {
    projected_mean = query_mean * key_mean[dimension];
    projected_variance =
        query_mean * query_mean * key_variance[dimension];
  }
  block_reduce_pair(projected_mean, projected_variance, reduction_scratch);
  const float threshold = projected_mean + threshold_sigma *
      sqrtf(fmaxf(projected_variance, 0.0f) + 1.0e-6f);

  const int route_word_count = (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
  if (linear_thread < route_word_count)
    shared_route[linear_thread] = 0;
  __syncthreads();

  const half *key_score_summary_head = key_score_summary +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) *
          padded_residual_summaries * HeadDim;
  const half *value_mean_head = value_mean +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) *
          padded_residual_summaries * HeadDim;
  // The initial INT8 Q tile lives in the second half of shared memory. Expand
  // it once into the first 16 KiB, which remains resident while routing and
  // skipped-block correction share the same Tensor Core scores.
  dequantize_int8_tile<HeadDim>(
      shared_initial_query_int8,
      query_scale_head,
      query_block * kBlockTokens,
      query_length,
      shared_correction_query);
  __syncthreads();

  // Route and approximate correction in one pass over 16 summaries. The
  // Tensor Core Q*K-centroid score supplies both per-token correction and the
  // centroid route score, eliminating a separate scalar scan of every K block.
  constexpr int residual_tokens = kBlockTokens / ResidualSubblocks;
  const int num_residual_summaries =
      (key_length + residual_tokens - 1) / residual_tokens;
  float *shared_proxy_partials = reinterpret_cast<float *>(
      shared_bytes + S::kRouteStorageOffset + kMaxRouteBytes);
  int summary_stage = 0;
  for (int summary_start = 0; summary_start < num_residual_summaries;
       summary_start += kSummaryTileTokens)
  {
    smem_t<SwizzleMode::k128B, G::kHalfPacks> current_summary_key(
        summary_stage == 0
            ? shared_summary_key.base
            : shared_summary_key_next.base);
    smem_t<SwizzleMode::k128B, G::kHalfPacks> current_summary_value(
        summary_stage == 0
            ? shared_summary_value.base
            : shared_summary_value_next.base);
    if constexpr (SparseValuePipeline)
    {
      if (summary_start == 0)
      {
        load_half_tile_async<HeadDim, kSummaryTileTokens>(
            key_score_summary_head,
            HeadDim,
            summary_start,
            num_residual_summaries,
            current_summary_key);
        load_half_tile_async<HeadDim, kSummaryTileTokens>(
            value_mean_head,
            HeadDim,
            summary_start,
            num_residual_summaries,
            current_summary_value);
        cp_async::commit_group();
        cp_async::wait_group<0>();
        __syncthreads();
      }
      const int next_summary_start = summary_start + kSummaryTileTokens;
      if (next_summary_start < num_residual_summaries)
      {
        smem_t<SwizzleMode::k128B, G::kHalfPacks> next_summary_key(
            summary_stage == 0
                ? shared_summary_key_next.base
                : shared_summary_key.base);
        smem_t<SwizzleMode::k128B, G::kHalfPacks> next_summary_value(
            summary_stage == 0
                ? shared_summary_value_next.base
                : shared_summary_value.base);
        load_half_tile_async<HeadDim, kSummaryTileTokens>(
            key_score_summary_head,
            HeadDim,
            next_summary_start,
            num_residual_summaries,
            next_summary_key);
        load_half_tile_async<HeadDim, kSummaryTileTokens>(
            value_mean_head,
            HeadDim,
            next_summary_start,
            num_residual_summaries,
            next_summary_value);
        cp_async::commit_group();
      }
    }
    else
    {
      load_half_tile_async<HeadDim, kSummaryTileTokens>(
          key_score_summary_head,
          HeadDim,
          summary_start,
          num_residual_summaries,
          current_summary_key);
      load_half_tile_async<HeadDim, kSummaryTileTokens>(
          value_mean_head,
          HeadDim,
          summary_start,
          num_residual_summaries,
          current_summary_value);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();
    }

    float score[1][1][8];
    compute_fp16_qk<HeadDim, 1>(
        shared_correction_query, current_summary_key, score);

    float proxy0 = score[0][0][0] + score[0][0][2];
    float proxy1 = score[0][0][1] + score[0][0][3];
    float proxy2 = score[0][0][4] + score[0][0][6];
    float proxy3 = score[0][0][5] + score[0][0][7];
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset >= 4; offset >>= 1)
    {
      proxy0 += __shfl_down_sync(0xffffffff, proxy0, offset);
      proxy1 += __shfl_down_sync(0xffffffff, proxy1, offset);
      proxy2 += __shfl_down_sync(0xffffffff, proxy2, offset);
      proxy3 += __shfl_down_sync(0xffffffff, proxy3, offset);
    }
    if (threadIdx.x < 4)
    {
      const int column_base = 2 * threadIdx.x;
      float *warp_proxy =
          shared_proxy_partials + threadIdx.y * kSummaryTileTokens;
      warp_proxy[column_base] = proxy0;
      warp_proxy[column_base + 1] = proxy1;
      warp_proxy[column_base + 8] = proxy2;
      warp_proxy[column_base + 9] = proxy3;
    }
    __syncthreads();

    constexpr int routed_blocks = kSummaryTileTokens / ResidualSubblocks;
    bool route_block = false;
    if (threadIdx.y == 0 && threadIdx.x < routed_blocks)
    {
      const int key_block =
          summary_start / ResidualSubblocks + threadIdx.x;
      if (key_block < active_key_blocks)
      {
        float proxy_sum = 0.0f;
        int key_token_count = 0;
#pragma unroll
        for (int residual_index = 0; residual_index < ResidualSubblocks;
             ++residual_index)
        {
          const int residual_summary =
              threadIdx.x * ResidualSubblocks + residual_index;
          const int residual_start =
              key_block * kBlockTokens + residual_index * residual_tokens;
          const int residual_count =
              max(0, min(residual_tokens, key_length - residual_start));
          float residual_proxy = 0.0f;
#pragma unroll
          for (int warp = 0; warp < kWarps; ++warp)
          {
            residual_proxy += shared_proxy_partials[
                warp * kSummaryTileTokens + residual_summary];
          }
          proxy_sum = fmaf(
              residual_proxy,
              static_cast<float>(residual_count),
              proxy_sum);
          key_token_count += residual_count;
        }
        const float proxy_score = proxy_sum /
            (static_cast<float>(query_token_count) * key_token_count);
        const int distance = query_block > key_block
            ? query_block - key_block
            : key_block - query_block;
        route_block = !sparse_query || exact_kv_blocks[key_block] ||
            distance <= 1 || proxy_score > threshold;
      }
    }
    if (threadIdx.y == 0)
    {
      const uint32_t route_bits = __ballot_sync(
          0xffffffffu, route_block);
      if (threadIdx.x == 0 && route_bits != 0)
      {
        const int first_key_block = summary_start / ResidualSubblocks;
        // Residual-1 emits aligned 16-bit ranges and residual-2 emits aligned
        // 8-bit ranges.  The summary loop is CTA-serial with a barrier below,
        // so exactly one lane owns this word update and no shared-memory
        // atomic is required.
        shared_route[first_key_block / kRouteWordBits] |=
            route_bits << (first_key_block % kRouteWordBits);
      }
    }
    __syncthreads();

#pragma unroll
    for (int element = 0; element < 8; ++element)
    {
      const int local_summary = 2 * (threadIdx.x % 4) +
          8 * (element / 4) + element % 2;
      const int residual_summary = summary_start + local_summary;
      const int key_block = residual_summary / ResidualSubblocks;
      const bool selected = key_block < active_key_blocks &&
          ((shared_route[key_block / kRouteWordBits] >>
            (key_block % kRouteWordBits)) & 1U);
      if (residual_summary >= num_residual_summaries || selected)
      {
        score[0][0][element] = -5000000.0f;
      }
      else
      {
        const int residual_index = residual_summary % ResidualSubblocks;
        const int residual_start =
            key_block * kBlockTokens + residual_index * residual_tokens;
        const int remaining = key_length - residual_start;
        const int block_length = remaining < residual_tokens
            ? remaining
            : residual_tokens;
        score[0][0][element] =
            score[0][0][element] * scale_log2 +
            math::ptx_log2(static_cast<float>(block_length));
      }
    }
    // W8A8 exact PV represents probabilities as U8 with an exp2 offset.
    // Keep skipped-block correction in that same online-softmax domain;
    // otherwise a later exact block compares a shifted maximum against an
    // unshifted one and rescales the correction by roughly 2^8.
    if constexpr (UseW8A8)
    {
      update_mdo<1, 1, G::kValueTiles, false, true, true>(
          score,
          output_fragment,
          row_max,
          denominator,
          1.0f,
          S_U8_OFFSET);
    }
    else
    {
      update_mdo<1, 1, G::kValueTiles, false, false, true>(
          score, output_fragment, row_max, denominator, 1.0f);
    }
    uint32_t probability[1][1][4];
    RS_32_to_16<1, 1>(score, probability);
    if constexpr (UseW8A8)
      accumulate_d<1, 1, ComputeUnit::kCudaCore>(score, denominator);
    else
      accumulate_d<1, 1, ComputeUnit::kTensorCore>(probability, denominator);
    uint32_t value_offset = value_mma_offset;
    compute_fp16_sv_permuted<4, 1, 1, 1, G::kValueTiles,
                             SwizzleMode::k128B, G::kHalfPacks, 4>(
        current_summary_value,
        probability,
        output_fragment,
        denominator,
        value_offset);
    __syncthreads();
    if constexpr (SparseValuePipeline)
    {
      if (summary_start + kSummaryTileTokens < num_residual_summaries)
      {
        cp_async::wait_group<0>();
        __syncthreads();
        summary_stage ^= 1;
      }
    }
  }

  __syncthreads();
  }
  }

  int compact_selected_count = 0;
  int register_selected_count = 0;
  RouteWords fp16_route{};
  if constexpr (!ForceDense)
  {
    if constexpr (UseW8A8)
    {
      compact_selected_count = compact_route_words<S::kSelectedCapacity>(
          shared_route,
          shared_selected_count,
          shared_selected_blocks,
          shared_compaction_scratch,
          active_key_blocks);
      if (selected_count != nullptr && sparse_query &&
          threadIdx.x == 0 && threadIdx.y == 0)
      {
        const unsigned long long count = static_cast<unsigned long long>(
          compact_selected_count >= 0
              ? compact_selected_count
              : -compact_selected_count);
        atomicAdd(selected_count, count);
      }
    }
    else
    {
      const int route_word_count =
          (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
      const int route_lane = threadIdx.x;
      fp16_route.word0 = route_lane < route_word_count
          ? shared_route[route_lane] : 0;
      fp16_route.word1 = route_lane + WARP_SIZE < route_word_count
          ? shared_route[route_lane + WARP_SIZE] : 0;
      fp16_route.word2 = route_lane + 2 * WARP_SIZE < route_word_count
          ? shared_route[route_lane + 2 * WARP_SIZE] : 0;
      fp16_route.word3 = route_lane + 3 * WARP_SIZE < route_word_count
          ? shared_route[route_lane + 3 * WARP_SIZE] : 0;
      unsigned int count = __popc(fp16_route.word0) +
          __popc(fp16_route.word1) + __popc(fp16_route.word2) +
          __popc(fp16_route.word3);
#pragma unroll
      for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xffffffff, count, offset);
      register_selected_count = static_cast<int>(
          __shfl_sync(0xffffffff, count, 0));
      if (selected_count != nullptr && sparse_query &&
          threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(
            selected_count,
            static_cast<unsigned long long>(register_selected_count));
    }
  }

  // Selected blocks retain exact token-level attention. Q/K are quantized once
  // with the production Sage per-16-row Q and per-64-row K scales, then use the
  // same SM75 INT8 Tensor Core MMA as stable Sage. V and output stay FP16/BF16
  // with FP32 online-softmax accumulation.
  load_int8_tile_async<HeadDim>(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_query_int8);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
  int selected_position = 0;
  int key_block = !ForceDense && sparse_query
      ? (UseW8A8
          ? next_compact_route_block<S::kSelectedCapacity>(
              shared_route,
              shared_selected_blocks,
              compact_selected_count,
              selected_position,
              0,
              active_key_blocks)
          : next_register_route_block(fp16_route, 0, active_key_blocks))
      : 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const bool value_ping_pong = UseW8A8 &&
      (ForceDense || !sparse_query || SparseValuePipeline);
#else
  constexpr bool value_ping_pong = false;
#endif
  int value_stage = 0;
  if (key_block < active_key_blocks)
  {
    load_int8_tile_async<HeadDim>(
        key_int8_head_ptr,
        stride_sequence_k_int8,
        key_block * kBlockTokens,
        key_length,
        shared_key_int8);
    if constexpr (UseW8A8)
    {
      load_quantized_value_tile_async<HeadDim>(
          value_int8_head_ptr,
          Varlen ? total_value_length : padded_value_length,
          key_block,
          shared_selected_value_int8);
      cp_async::commit_group();
    }
    else
    {
      cp_async::commit_group();
      if constexpr (MappedValue)
      {
        load_half_tile_mapped<HeadDim, kBlockTokens>(
            value_head_ptr,
            stride_sequence_v,
            value_source_indices,
            key_block * kBlockTokens,
            key_length,
            shared_selected_value);
      }
      else
      {
        load_half_tile<HeadDim, kBlockTokens>(
            value_head_ptr,
            stride_sequence_v,
            key_block * kBlockTokens,
            key_length,
            shared_selected_value);
      }
    }
    cp_async::wait_group<0>();
    __syncthreads();
  }
  while (key_block < active_key_blocks)
  {
    // Integer QK accumulators are dead before online softmax consumes the
    // converted values.  Make that lifetime overlap explicit so nvcc does not
    // reserve two independent 32-register score fragments on long D128
    // instantiations.
    union ScoreStorage
    {
      int32_t integer[1][4][8];
      float floating[1][4][8];
    } score_storage;
    compute_int8_qk<HeadDim>(
        shared_query_int8, shared_key_int8, score_storage.integer);
#pragma unroll
    for (int key_tile = 0; key_tile < 4; ++key_tile)
    {
#pragma unroll
      for (int element = 0; element < 8; ++element)
        score_storage.floating[0][key_tile][element] = __int2float_rz(
            score_storage.integer[0][key_tile][element]);
    }
    float (&score)[1][4][8] = score_storage.floating;
    const int next_key_block = !ForceDense && sparse_query
        ? (UseW8A8
            ? next_compact_route_block<S::kSelectedCapacity>(
                shared_route,
                shared_selected_blocks,
                compact_selected_count,
                ++selected_position,
                key_block + 1,
                active_key_blocks)
            : next_register_route_block(
                fp16_route, key_block + 1, active_key_blocks))
        : key_block + 1;
    const bool has_next = next_key_block < active_key_blocks;
    __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (has_next)
    {
      load_int8_tile_async<HeadDim>(
          key_int8_head_ptr,
          stride_sequence_k_int8,
          next_key_block * kBlockTokens,
          key_length,
          shared_key_int8);
      if constexpr (UseW8A8)
      {
        if (value_ping_pong)
        {
          smem_t<SwizzleMode::k64B, 4> next_value(
              value_stage == 0
                  ? shared_selected_value_int8_next.base
                  : shared_selected_value_int8.base);
          load_quantized_value_tile_async<HeadDim>(
              value_int8_head_ptr,
              Varlen ? total_value_length : padded_value_length,
              next_key_block,
              next_value);
        }
      }
      cp_async::commit_group();
    }
#endif
    const uint32_t key_lane_base =
        key_block * kBlockTokens + 2 * (threadIdx.x % 4);
    apply_out_of_bound_mask<1, 4>(key_lane_base, score, key_length);
    if constexpr (ExternalRoute) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int key = key_lane_base + kt * 16 + (e / 4) * 8 + (e & 1);
          if (key < key_length && !value_source_indices[key]) score[0][kt][e] = -INFINITY;
        }
      }
    }
    if constexpr (IsCausal)
    {
      const uint32_t query_lane_base = query_block * kBlockTokens +
          threadIdx.y * 16 + threadIdx.x / 4;
      apply_causal_mask<1, 4>(query_lane_base, key_lane_base, score);
    }
    if constexpr (UseW8A8)
    {
      update_mdo<1, 4, G::kValueTiles, false, true, false>(
          score,
          output_fragment,
          row_max,
          denominator,
          scale_log2 * q_dequant_scale * key_scale_head[key_block],
          S_U8_OFFSET);
      uint32_t probability_u8[1][2][4];
      RS_to_u8<1, 4>(score, probability_u8);
      accumulate_d<1, 4, ComputeUnit::kCudaCore>(score, denominator);
      float probability_scale[1][2] = {{1.0f, 1.0f}};
      smem_t<SwizzleMode::k64B, 4> current_value(
          value_stage == 0
              ? shared_selected_value_int8.base
              : shared_selected_value_int8_next.base);
      compute_int8_sv_permuted<1, 4, G::kValueTiles, SwizzleMode::k64B, 4>(
          current_value,
          probability_scale,
          probability_u8,
          output_fragment);
    }
    else
    {
      update_mdo<1, 4, G::kValueTiles, false, false, false>(
          score,
          output_fragment,
          row_max,
          denominator,
          scale_log2 * q_dequant_scale * key_scale_head[key_block]);
      uint32_t probability[1][4][4];
      RS_32_to_16<1, 4>(score, probability);
      accumulate_d<1, 4, ComputeUnit::kTensorCore>(probability, denominator);
      uint32_t value_offset = value_mma_offset;
      compute_fp16_sv_permuted<4, 1, 1, 4, G::kValueTiles,
                               SwizzleMode::k128B, G::kHalfPacks, 4>(
          shared_selected_value,
          probability,
          output_fragment,
          denominator,
                               value_offset);
    }
    __syncthreads();
    if (has_next)
    {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
      load_int8_tile_async<HeadDim>(
          key_int8_head_ptr,
          stride_sequence_k_int8,
          next_key_block * kBlockTokens,
          key_length,
          shared_key_int8);
      cp_async::commit_group();
#endif
      if constexpr (UseW8A8)
      {
        if (!value_ping_pong)
        {
          load_quantized_value_tile_async<HeadDim>(
              value_int8_head_ptr,
              Varlen ? total_value_length : padded_value_length,
              next_key_block,
              shared_selected_value_int8);
          cp_async::commit_group();
        }
      }
      else
      {
        if constexpr (MappedValue)
        {
          load_half_tile_mapped<HeadDim, kBlockTokens>(
              value_head_ptr,
              stride_sequence_v,
              value_source_indices,
              next_key_block * kBlockTokens,
              key_length,
              shared_selected_value);
        }
        else
        {
          load_half_tile<HeadDim, kBlockTokens>(
              value_head_ptr,
              stride_sequence_v,
              next_key_block * kBlockTokens,
              key_length,
              shared_selected_value);
        }
      }
      cp_async::wait_group<0>();
      __syncthreads();
      if (value_ping_pong)
        value_stage ^= 1;
    }
    key_block = next_key_block;
  }

  if constexpr (UseW8A8)
  {
    normalize_d<1, G::kValueTiles, ComputeUnit::kCudaCore>(
        output_fragment, row_max, denominator);
    float channel_scale[4];
    const float *scale_base = value_scale_head + (threadIdx.x % 4) * 2;
#pragma unroll
    for (int value_tile = 0; value_tile < G::kValueTiles; ++value_tile)
    {
      reinterpret_cast<float2 *>(channel_scale)[0] =
          *reinterpret_cast<const float2 *>(scale_base + value_tile * 16);
      reinterpret_cast<float2 *>(channel_scale)[1] =
          *reinterpret_cast<const float2 *>(scale_base + value_tile * 16 + 8);
#pragma unroll
      for (int element = 0; element < 8; ++element)
      {
        output_fragment[0][value_tile][element] *=
            channel_scale[(element / 4) * 2 + (element % 2)];
      }
    }
  }
  else
  {
    normalize_d<1, G::kValueTiles, ComputeUnit::kTensorCore>(
        output_fragment, row_max, denominator);
  }

  const uint32_t output_row_base = threadIdx.y * 16 + threadIdx.x / 4;
#pragma unroll
  for (int value_tile = 0; value_tile < G::kValueTiles; ++value_tile)
  {
    const uint32_t output_offset = shared_output.get_permuted_offset(
        output_row_base, value_tile * 2);
    uint32_t converted[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
    {
      if constexpr (std::is_same<T, half>::value)
      {
        reinterpret_cast<half2 *>(converted)[pair] =
            __float22half2_rn(reinterpret_cast<float2 *>(output_fragment[0][value_tile])[pair]);
      }
      else
      {
        reinterpret_cast<nv_bfloat162 *>(converted)[pair] =
            __float22bfloat162_rn(
                reinterpret_cast<float2 *>(output_fragment[0][value_tile])[pair]);
      }
    }
    reinterpret_cast<uint32_t *>(shared_output.base + output_offset)[threadIdx.x % 4] =
        converted[0];
    reinterpret_cast<uint32_t *>(
        shared_output.base + output_offset + 8 * G::kHalfPacks)[threadIdx.x % 4] =
        converted[1];
    reinterpret_cast<uint32_t *>(shared_output.base + (output_offset ^ 0x1))[threadIdx.x % 4] =
        converted[2];
    reinterpret_cast<uint32_t *>(
        shared_output.base + (output_offset ^ 0x1) + 8 * G::kHalfPacks)[threadIdx.x % 4] =
        converted[3];
  }
  __syncthreads();

  constexpr int output_line_lanes = 8;
  constexpr int output_rows_per_warp = 4;
  constexpr int output_column_groups = HeadDim / 64;
  T *output_lane = output_head_ptr +
      (query_block * kBlockTokens + threadIdx.y * 16 +
       threadIdx.x / output_line_lanes) *
          stride_sequence_o +
      (threadIdx.x % output_line_lanes) * 8;
  uint32_t output_offset = shared_output.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x / output_line_lanes,
      threadIdx.x % output_line_lanes);
  int output_row = query_block * kBlockTokens + threadIdx.y * 16 +
      threadIdx.x / output_line_lanes;
#pragma unroll
  for (int row_group = 0; row_group < 4; ++row_group)
  {
#pragma unroll
    for (int column_group = 0; column_group < output_column_groups; ++column_group)
    {
      if (output_row < query_length)
        shared_output.store_128b(output_offset, output_lane);
      output_lane += output_line_lanes * 8;
      output_offset = shared_output.advance_offset_by_column<8>(output_offset);
    }
    output_offset = shared_output.advance_offset_by_row<output_rows_per_warp>(
        output_offset - output_column_groups * output_line_lanes);
    output_lane += output_rows_per_warp * stride_sequence_o -
        output_column_groups * output_line_lanes * 8;
    output_row += output_rows_per_warp;
  }
}

void check_launch(const char *name)
{
  const cudaError_t error = cudaGetLastError();
  VEDA_CHECK(error == cudaSuccess, name, " launch failed: ", cudaGetErrorString(error));
}

int current_cuda_device_major()
{
  int device = 0;
  const cudaError_t device_error = cudaGetDevice(&device);
  VEDA_CHECK(
      device_error == cudaSuccess,
      "unable to query the current CUDA device: ",
      cudaGetErrorString(device_error));
  // Attention dispatch runs repeatedly on the same ComfyUI worker thread.
  // Cache only the immutable capability, while still observing current-device
  // changes when a process alternates between its Turing and Ampere cards.
  static thread_local int cached_device = -1;
  static thread_local int cached_major = 0;
  if (device == cached_device)
    return cached_major;
  int device_major = 0;
  const cudaError_t capability_error = cudaDeviceGetAttribute(
      &device_major, cudaDevAttrComputeCapabilityMajor, device);
  VEDA_CHECK(
      capability_error == cudaSuccess,
      "unable to query the current CUDA capability: ",
      cudaGetErrorString(capability_error));
  cached_device = device;
  cached_major = device_major;
  return cached_major;
}

} // namespace
