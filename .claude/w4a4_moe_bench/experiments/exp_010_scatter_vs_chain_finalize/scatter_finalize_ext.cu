#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kHidden = 2048;
constexpr int kOutputTiles = kHidden / kTileN;
constexpr int kThreads = 160;
constexpr int kComputeWarps = 4;
constexpr int kSharedBytes = 92160;

__device__ __forceinline__ uint64_t read_globaltimer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void compute_warp_barrier() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

__device__ __forceinline__ float canonical_value(int physical_row, int slice, int column) {
  // Power-of-two scale keeps the diagnostic numerically quiet while retaining
  // nonzero, row/slice/column-dependent BF16 values.
  int code = (physical_row * 3 + slice * 5 + column) & 7;
  return static_cast<float>(code + 1) * 0.0009765625f;  // 2^-10
}

__device__ __forceinline__ void red_v4_bf16x2(__nv_bfloat16* address, float v0, float v1, float v2,
                                              float v3, float v4, float v5, float v6, float v7) {
  asm volatile(
      "{ .reg .b32 p0,p1,p2,p3;"
      " cvt.rn.satfinite.bf16x2.f32 p0, %2, %1;"
      " cvt.rn.satfinite.bf16x2.f32 p1, %4, %3;"
      " cvt.rn.satfinite.bf16x2.f32 p2, %6, %5;"
      " cvt.rn.satfinite.bf16x2.f32 p3, %8, %7;"
      " red.global.add.noftz.v4.bf16x2 [%0], {p0,p1,p2,p3}; }"
      :
      : "l"(address), "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7)
      : "memory");
}

__global__ void fill_partials_kernel(__nv_bfloat16* partials, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    int column = index % kHidden;
    int64_t row_slice = index / kHidden;
    int slice = row_slice & 3;
    int physical_row = static_cast<int>(row_slice >> 2);
    partials[index] = __float2bfloat16_rn(canonical_value(physical_row, slice, column));
  }
}

__global__ void fill_full_rows_kernel(__nv_bfloat16* rows, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    int column = index % kHidden;
    int physical_row = static_cast<int>(index / kHidden);
    float value = 0.0f;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
      value += canonical_value(physical_row, slice, column);
    }
    rows[index] = __float2bfloat16_rn(value);
  }
}

__device__ __forceinline__ int clamp_warp_rows(int valid_rows, int warp_m_base) {
  int rows = valid_rows - warp_m_base;
  return rows < 0 ? 0 : (rows > 64 ? 64 : rows);
}

__global__ void scatter_kernel(
    const __nv_bfloat16* __restrict__ partials, const int32_t* __restrict__ token_map,
    const float* __restrict__ token_weights, const int32_t* __restrict__ route_slot,
    const int32_t* __restrict__ cta_offsets, const int32_t* __restrict__ cta_tasks,
    const int32_t* __restrict__ task_m_tile, const int32_t* __restrict__ task_slice,
    const int32_t* __restrict__ task_valid_rows, const int64_t* __restrict__ cadence_ns,
    __nv_bfloat16* __restrict__ output, int64_t* __restrict__ timestamps, int task_count,
    int num_tokens, int shard_factor, int mode, int span_matched, int direct_grid) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* sC = reinterpret_cast<__nv_bfloat16*>(shared_raw);
  auto* sTok = reinterpret_cast<int32_t*>(shared_raw + kTileM * kTileN * 2);
  auto* sWeight =
      reinterpret_cast<float*>(shared_raw + kTileM * kTileN * 2 + kTileM * sizeof(int32_t));

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int begin = direct_grid ? static_cast<int>(blockIdx.x) : cta_offsets[blockIdx.x];
  int end = direct_grid ? begin + 1 : cta_offsets[blockIdx.x + 1];

  for (int queue_index = begin; queue_index < end; ++queue_index) {
    int task = direct_grid ? begin : cta_tasks[queue_index];
    if (task >= task_count) {
      continue;
    }
    int tile_m = task_m_tile[task];
    int slice = task_slice[task];
    int valid_rows = task_valid_rows[task];

    for (int row = tid; row < kTileM; row += kThreads) {
      int physical_row = tile_m * kTileM + row;
      if (row < valid_rows) {
        sTok[row] = token_map[physical_row];
        sWeight[row] = token_weights[physical_row];
      } else {
        sTok[row] = 0;
        sWeight[row] = 0.0f;
      }
    }
    __syncthreads();

    for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
      for (int element = tid; element < kTileM * kTileN; element += kThreads) {
        int local_row = element / kTileN;
        int local_col = element - local_row * kTileN;
        int physical_row = tile_m * kTileM + local_row;
        int global_col = output_tile * kTileN + local_col;
        float value = 0.0f;
        if (local_row < valid_rows) {
          if (mode == 1) {
            int64_t offset =
                (static_cast<int64_t>(physical_row) * 4 + slice) * kHidden + global_col;
            value = __bfloat162float(partials[offset]);
          } else {
            value = canonical_value(physical_row, slice, global_col);
          }
        }
        sC[element] = __float2bfloat16_rn(value);
      }
      __syncthreads();

      if (mode == 0 && cadence_ns != nullptr) {
        if (tid == 0) {
          uint64_t start = read_globaltimer();
          uint64_t wait = static_cast<uint64_t>(
              cadence_ns[static_cast<int64_t>(task) * kOutputTiles + output_tile]);
          while (read_globaltimer() - start < wait) {
          }
        }
        __syncthreads();
      }

      if (warp < kComputeWarps) {
        compute_warp_barrier();
        if (lane == 0) {
          int64_t base =
              (((static_cast<int64_t>(task) * kOutputTiles + output_tile) * kComputeWarps + warp) *
               3);
          timestamps[base] = static_cast<int64_t>(read_globaltimer());
        }
        asm volatile("" ::: "memory");

        int warp_m_base = (warp >> 1) * 64;
        int warp_n_base = (warp & 1) * 64;
        int warp_rows = clamp_warp_rows(valid_rows, warp_m_base);
        int vec_idx = lane;
        while (vec_idx < warp_rows * 8) {
          int local_row = vec_idx >> 3;
          int local_vec_col = vec_idx & 7;
          int local_col = warp_n_base + local_vec_col * 8;
          int cached_row = warp_m_base + local_row;
          int token = sTok[cached_row];
          float weight = sWeight[cached_row];
          int physical_row = tile_m * kTileM + cached_row;
          int shard = 0;
          if (span_matched != 0) {
            shard = token & 31;
          } else if (shard_factor > 1) {
            int logical_contribution = route_slot[physical_row] * 4 + slice;
            shard = logical_contribution % shard_factor;
          }
          int global_col = output_tile * kTileN + local_col;
          __nv_bfloat16* destination =
              output + (static_cast<int64_t>(shard) * num_tokens + token) * kHidden + global_col;
          int smem_base = cached_row * kTileN + local_col;
          float v0 = weight * __bfloat162float(sC[smem_base + 0]);
          float v1 = weight * __bfloat162float(sC[smem_base + 1]);
          float v2 = weight * __bfloat162float(sC[smem_base + 2]);
          float v3 = weight * __bfloat162float(sC[smem_base + 3]);
          float v4 = weight * __bfloat162float(sC[smem_base + 4]);
          float v5 = weight * __bfloat162float(sC[smem_base + 5]);
          float v6 = weight * __bfloat162float(sC[smem_base + 6]);
          float v7 = weight * __bfloat162float(sC[smem_base + 7]);
          red_v4_bf16x2(destination, v0, v1, v2, v3, v4, v5, v6, v7);
          vec_idx += 32;
        }

        if (lane == 0) {
          int64_t base =
              (((static_cast<int64_t>(task) * kOutputTiles + output_tile) * kComputeWarps + warp) *
               3);
          timestamps[base + 1] = static_cast<int64_t>(read_globaltimer());
        }
        compute_warp_barrier();
        if (lane == 0) {
          int64_t base =
              (((static_cast<int64_t>(task) * kOutputTiles + output_tile) * kComputeWarps + warp) *
               3);
          timestamps[base + 2] = static_cast<int64_t>(read_globaltimer());
        }
      }
      __syncthreads();
    }
  }
}

__global__ void chain_finalize_kernel(const __nv_bfloat16* __restrict__ expanded_rows,
                                      const int32_t* __restrict__ route_rows,
                                      const float* __restrict__ scales,
                                      __nv_bfloat16* __restrict__ output, int num_tokens) {
  int token = blockIdx.x;
  int column = threadIdx.x * 8;
  if (token >= num_tokens || column >= kHidden) {
    return;
  }
  float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int route = 0; route < 8; ++route) {
    int physical_row = route_rows[token * 8 + route];
    float scale = scales[token * 8 + route];
    const __nv_bfloat16* source =
        expanded_rows + static_cast<int64_t>(physical_row) * kHidden + column;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      accumulator[element] += scale * __bfloat162float(source[element]);
    }
  }
  __nv_bfloat16* destination = output + static_cast<int64_t>(token) * kHidden + column;
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    destination[element] = __float2bfloat16_rn(accumulator[element]);
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void fill_partials(torch::Tensor partials) {
  check_cuda_contiguous(partials, "partials");
  TORCH_CHECK(partials.scalar_type() == torch::kBFloat16, "partials must be BF16");
  int blocks = std::min<int64_t>(65535, (partials.numel() + 255) / 256);
  fill_partials_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(partials.data_ptr()), partials.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_full_rows(torch::Tensor rows) {
  check_cuda_contiguous(rows, "rows");
  TORCH_CHECK(rows.scalar_type() == torch::kBFloat16, "rows must be BF16");
  int blocks = std::min<int64_t>(65535, (rows.numel() + 255) / 256);
  fill_full_rows_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(rows.data_ptr()), rows.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_launch(torch::Tensor partials, torch::Tensor token_map, torch::Tensor token_weights,
                    torch::Tensor route_slot, torch::Tensor cta_offsets, torch::Tensor cta_tasks,
                    torch::Tensor task_m_tile, torch::Tensor task_slice,
                    torch::Tensor task_valid_rows, torch::Tensor cadence_ns, torch::Tensor output,
                    torch::Tensor timestamps, int64_t task_count, int64_t num_tokens,
                    int64_t shard_factor, int64_t mode, bool span_matched, bool direct_grid) {
  for (const auto& pair :
       std::vector<std::pair<torch::Tensor, const char*>>{{partials, "partials"},
                                                          {token_map, "token_map"},
                                                          {token_weights, "token_weights"},
                                                          {route_slot, "route_slot"},
                                                          {cta_offsets, "cta_offsets"},
                                                          {cta_tasks, "cta_tasks"},
                                                          {task_m_tile, "task_m_tile"},
                                                          {task_slice, "task_slice"},
                                                          {task_valid_rows, "task_valid_rows"},
                                                          {cadence_ns, "cadence_ns"},
                                                          {output, "output"},
                                                          {timestamps, "timestamps"}}) {
    check_cuda_contiguous(pair.first, pair.second);
  }
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be BF16");
  TORCH_CHECK(timestamps.scalar_type() == torch::kInt64, "timestamps must be int64");
  TORCH_CHECK(task_count > 0 && task_count <= task_m_tile.numel(), "invalid task_count");
  TORCH_CHECK(shard_factor == 1 || shard_factor == 4 || shard_factor == 32,
              "shard_factor must be 1, 4, or 32");
  int blocks =
      direct_grid ? static_cast<int>(task_count) : static_cast<int>(cta_offsets.numel() - 1);
  auto kernel = scatter_kernel;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  scatter_kernel<<<blocks, kThreads, kSharedBytes, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(partials.data_ptr()), token_map.data_ptr<int32_t>(),
      token_weights.data_ptr<float>(), route_slot.data_ptr<int32_t>(),
      cta_offsets.data_ptr<int32_t>(), cta_tasks.data_ptr<int32_t>(),
      task_m_tile.data_ptr<int32_t>(), task_slice.data_ptr<int32_t>(),
      task_valid_rows.data_ptr<int32_t>(), cadence_ns.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), timestamps.data_ptr<int64_t>(),
      static_cast<int>(task_count), static_cast<int>(num_tokens), static_cast<int>(shard_factor),
      static_cast<int>(mode), span_matched ? 1 : 0, direct_grid ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void chain_finalize(torch::Tensor expanded_rows, torch::Tensor route_rows, torch::Tensor scales,
                    torch::Tensor output) {
  check_cuda_contiguous(expanded_rows, "expanded_rows");
  check_cuda_contiguous(route_rows, "route_rows");
  check_cuda_contiguous(scales, "scales");
  check_cuda_contiguous(output, "output");
  TORCH_CHECK(expanded_rows.scalar_type() == torch::kBFloat16, "expanded_rows must be BF16");
  TORCH_CHECK(route_rows.scalar_type() == torch::kInt32, "route_rows must be int32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be BF16");
  int num_tokens = output.size(0);
  chain_finalize_kernel<<<num_tokens, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(expanded_rows.data_ptr()),
      route_rows.data_ptr<int32_t>(), scales.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), num_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fill_partials", &fill_partials);
  module.def("fill_full_rows", &fill_full_rows);
  module.def("scatter_launch", &scatter_launch);
  module.def("chain_finalize", &chain_finalize);
}
