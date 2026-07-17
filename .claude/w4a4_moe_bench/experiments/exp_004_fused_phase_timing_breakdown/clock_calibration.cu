#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

__device__ __forceinline__ std::uint64_t read_clock64() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value));
  return value;
}

__global__ void clock_store_pair_kernel(std::uint64_t* output, std::int64_t samples) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= 5 || lane != 0) {
    return;
  }
  const std::int64_t warp_base = static_cast<std::int64_t>(warp) * samples * 2;
  for (std::int64_t sample = 0; sample < samples; ++sample) {
    const std::uint64_t begin = read_clock64();
    output[warp_base + sample * 2] = begin;
    const std::uint64_t end = read_clock64();
    output[warp_base + sample * 2 + 1] = end;
  }
}

}  // namespace

torch::Tensor exp004_clock_store_pairs(std::int64_t samples) {
  TORCH_CHECK(samples > 0, "samples must be positive");
  auto output = torch::empty({5, samples, 2},
                             torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
  const auto stream = at::cuda::getCurrentCUDAStream();
  clock_store_pair_kernel<<<1, 160, 0, stream>>>(
      reinterpret_cast<std::uint64_t*>(output.data_ptr<std::int64_t>()), samples);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("clock_store_pairs", &exp004_clock_store_pairs,
             "Back-to-back clock64 + lane-0 GMEM-store calibration");
}
