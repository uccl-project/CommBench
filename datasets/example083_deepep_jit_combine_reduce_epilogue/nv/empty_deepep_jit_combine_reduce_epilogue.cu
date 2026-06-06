#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/layout.cuh>

#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleoutRanks,
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* recv_buffer,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    EP_STATIC_ASSERT(kNumExperts % (kNumScaleupRanks * kNumScaleoutRanks) == 0, "Invalid number of experts or ranks");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

    // Will block until the main combine kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();

    // TODO: Implement DeepEP's combine reduce epilogue.
    // Publish deterministic zero weights while leaving the main result missing.
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {
        if (combined_topk_weights != nullptr) {
            if (lane_idx < kNumTopk)
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = 0.0f;
        }
        __syncwarp();
    }
}

}  // deep_ep::elastic
