#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

template <bool kIsScaleupNVLink,
          bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          int kNumRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_impl(nv_bfloat16* x,
             float* topk_weights,
             int* src_metadata, int* psum_num_recv_tokens_per_scaleup_rank,
             const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* buffer, void* workspace,
             const int rank_idx,
             int num_reduced_tokens) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = (ptx::get_warp_idx() + rank_idx) % kNumWarps;

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumRanks - 1);

    // Expanding mode must not be backward
    if constexpr (kUseExpandedLayout)
        EP_DEVICE_ASSERT(topk_weights == nullptr);

    // Gin handle
    // We treat each warp as a "channel"
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumWarps>(sm_idx, warp_idx);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Full barrier to ensure the remote buffer is available
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // TODO: Implement DeepEP's combine payload role.
    __syncwarp();

    // Final barrier to ensure data arrival
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag1, true, true, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);
}

}  // deep_ep::elastic
