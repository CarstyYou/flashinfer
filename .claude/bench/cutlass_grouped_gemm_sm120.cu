// cutlass_grouped_gemm_sm120.cu — SM120 FP8 grouped GEMM, 1D per-token UE8M0/FP32 scale.
//
// Stripped from CUTLASS example 87c (87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise),
// changed ScaleGranularityN=1 for 1D per-token scale (apples-to-apples with cute/cudnn/DG).
// Two variants compiled via template: granK=128 (TileK=128) and granK=32 (TileK=32).

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

template <int TileM_, int TileK_, int ScaleGranK_>
struct Traits {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 16;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 16;

  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 8;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  static constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<Int<TileM_>, _128, Int<TileK_>>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  using ElementSF = ElementAccumulator;
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 1, ScaleGranK_>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  // Cooperative schedule requires TileM >= 128. Pingpong allows TileM >= 64.
  using Schedule =
      cute::conditional_t<(TileM_ >= 128), cutlass::gemm::KernelScheduleSm120Blockwise,
                          cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB, cute::tuple<LayoutB*, LayoutSFB*>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      Schedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
};

template <int TileM_, int TileK_, int ScaleGranK_>
torch::Tensor run_impl(
    torch::Tensor a,        // (cum_m, k) FP8 e4m3
    torch::Tensor b,        // (E, n, k) FP8 e4m3
    torch::Tensor a_scale,  // FP32 (E, k_blocks, m_pe) per-group M-major contiguous
    torch::Tensor b_scale,  // FP32 (E, k_blocks, n) per-group N-major contiguous
    torch::Tensor m_indptr  // (E+1,) int32
) {
  using T = Traits<TileM_, TileK_, ScaleGranK_>;
  using Gemm = typename T::Gemm;
  using ElementA = typename T::ElementA;
  using ElementB = typename T::ElementB;
  using ElementC = typename T::ElementC;
  using ElementD = typename T::ElementD;
  using ElementSF = typename T::ElementSF;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using LayoutSFA = typename T::LayoutSFA;
  using LayoutSFB = typename T::LayoutSFB;
  using ScaleConfig = typename T::ScaleConfig;
  using ProblemShape = typename T::ProblemShape;

  TORCH_CHECK(m_indptr.dtype() == torch::kInt32, "m_indptr must be int32");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must be on cuda");

  auto m_indptr_cpu = m_indptr.cpu();
  auto m_data = m_indptr_cpu.data_ptr<int32_t>();
  int num_groups = static_cast<int>(m_indptr.size(0)) - 1;
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(b.size(2));
  int k_blocks = k / ScaleGranK_;

  auto out = torch::empty({a.size(0), n}, torch::dtype(torch::kBFloat16).device(a.device()));

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes(num_groups);
  std::vector<ElementA const*> ptr_A(num_groups);
  std::vector<ElementB const*> ptr_B(num_groups);
  std::vector<ElementSF const*> ptr_SFA(num_groups);
  std::vector<ElementSF const*> ptr_SFB(num_groups);
  std::vector<ElementD*> ptr_D(num_groups);
  std::vector<StrideA> stride_A(num_groups);
  std::vector<StrideB> stride_B(num_groups);
  std::vector<StrideD> stride_D(num_groups);
  std::vector<LayoutSFA> layout_SFA(num_groups);
  std::vector<LayoutSFB> layout_SFB(num_groups);

  ElementA* a_base = reinterpret_cast<ElementA*>(a.data_ptr());
  ElementB* b_base = reinterpret_cast<ElementB*>(b.data_ptr());
  ElementD* d_base = reinterpret_cast<ElementD*>(out.data_ptr());
  ElementSF* sfa_base = reinterpret_cast<ElementSF*>(a_scale.data_ptr());
  ElementSF* sfb_base = reinterpret_cast<ElementSF*>(b_scale.data_ptr());

  for (int i = 0; i < num_groups; ++i) {
    int m_start = m_data[i];
    int m_end = m_data[i + 1];
    int m_grp = m_end - m_start;

    problem_sizes[i] = {m_grp, n, k};
    stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {m_grp, k, 1});
    stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {m_grp, n, 1});
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m_grp, n, k, 1));
    layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m_grp, n, k, 1));

    ptr_A[i] = a_base + static_cast<size_t>(m_start) * k;
    ptr_B[i] = b_base + static_cast<size_t>(i) * n * k;
    ptr_D[i] = d_base + static_cast<size_t>(m_start) * n;
    ptr_SFA[i] = sfa_base + static_cast<size_t>(i) * m_grp * k_blocks;
    ptr_SFB[i] = sfb_base + static_cast<size_t>(i) * n * k_blocks;
  }

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes_d(
      num_groups);
  cutlass::DeviceAllocation<ElementA const*> ptr_A_d(num_groups);
  cutlass::DeviceAllocation<ElementB const*> ptr_B_d(num_groups);
  cutlass::DeviceAllocation<ElementSF const*> ptr_SFA_d(num_groups);
  cutlass::DeviceAllocation<ElementSF const*> ptr_SFB_d(num_groups);
  cutlass::DeviceAllocation<ElementD*> ptr_D_d(num_groups);
  cutlass::DeviceAllocation<StrideA> stride_A_d(num_groups);
  cutlass::DeviceAllocation<StrideB> stride_B_d(num_groups);
  cutlass::DeviceAllocation<StrideD> stride_D_d(num_groups);
  cutlass::DeviceAllocation<LayoutSFA> layout_SFA_d(num_groups);
  cutlass::DeviceAllocation<LayoutSFB> layout_SFB_d(num_groups);

  problem_sizes_d.copy_from_host(problem_sizes.data());
  ptr_A_d.copy_from_host(ptr_A.data());
  ptr_B_d.copy_from_host(ptr_B.data());
  ptr_SFA_d.copy_from_host(ptr_SFA.data());
  ptr_SFB_d.copy_from_host(ptr_SFB.data());
  ptr_D_d.copy_from_host(ptr_D.data());
  stride_A_d.copy_from_host(stride_A.data());
  stride_B_d.copy_from_host(stride_B.data());
  stride_D_d.copy_from_host(stride_D.data());
  layout_SFA_d.copy_from_host(layout_SFA.data());
  layout_SFB_d.copy_from_host(layout_SFB.data());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device().index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  arguments = {cutlass::gemm::GemmUniversalMode::kGrouped,
               {num_groups, problem_sizes_d.get(), problem_sizes.data()},
               {ptr_A_d.get(), stride_A_d.get(), ptr_B_d.get(), stride_B_d.get(), ptr_SFA_d.get(),
                layout_SFA_d.get(), ptr_SFB_d.get(), layout_SFB_d.get()},
               {fusion_args, nullptr, nullptr, ptr_D_d.get(), stride_D_d.get()},
               hw_info,
               scheduler};

  Gemm gemm;
  auto ws_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> ws(ws_size);

  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "cutlass gemm.can_implement failed: ", cutlass::cutlassGetStatusString(status));

  status = gemm.initialize(arguments, ws.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "cutlass gemm.initialize failed: ", cutlass::cutlassGetStatusString(status));

  auto stream = at::cuda::getCurrentCUDAStream();
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "cutlass gemm.run failed: ", cutlass::cutlassGetStatusString(status));

  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_m128_k128", &run_impl<128, 128, 128>,
        "SM120 grouped GEMM, TileM=128 Cooperative TileK=128 granK=128 (large m_pe)", py::arg("a"),
        py::arg("b"), py::arg("a_scale"), py::arg("b_scale"), py::arg("m_indptr"));
  m.def("run_m128_k32", &run_impl<128, 32, 32>,
        "SM120 grouped GEMM, TileM=128 Cooperative TileK=32 granK=32 (large m_pe)", py::arg("a"),
        py::arg("b"), py::arg("a_scale"), py::arg("b_scale"), py::arg("m_indptr"));
  m.def("run_m64_k128", &run_impl<64, 128, 128>,
        "SM120 grouped GEMM, TileM=64 Pingpong TileK=128 granK=128 (small m_pe)", py::arg("a"),
        py::arg("b"), py::arg("a_scale"), py::arg("b_scale"), py::arg("m_indptr"));
  m.def("run_m64_k32", &run_impl<64, 32, 32>,
        "SM120 grouped GEMM, TileM=64 Pingpong TileK=32 granK=32 (small m_pe)", py::arg("a"),
        py::arg("b"), py::arg("a_scale"), py::arg("b_scale"), py::arg("m_indptr"));
}
