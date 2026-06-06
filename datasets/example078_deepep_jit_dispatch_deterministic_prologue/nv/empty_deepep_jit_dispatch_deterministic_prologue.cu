#pragma once

#include <cooperative_groups.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <int kNumSMs, int kNumWarps,
          int kNumScaleupRanks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_deterministic_prologue_impl(
    topk_idx_t* topk_idx,
    int* rank_count_buffer,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int scaleup_rank_idx
) {
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");

    // TODO: Implement the complete deterministic dispatch prologue kernel.
    //
    // Required functionality:
    // - Read topk_idx[num_tokens, kNumTopk] and map every valid expert id to
    //   its destination scale-up rank. Invalid selections are marked with -1.
    // - For each token, count a destination rank only once even if multiple
    //   top-k experts belong to that same rank.
    // - Fill rank_count_buffer[sm, rank] with the number of deduplicated token
    //   routes contributed by each SM for each scale-up rank.
    // - Assign deterministic destination slots for every valid token/rank
    //   route, using all earlier SMs and earlier work within the current SM as
    //   the prefix for that rank.
    // - Write dst_buffer_slot_idx[token, topk] as
    //   scaleup_rank_idx * kNumMaxTokensPerRank + local_slot for valid
    //   deduplicated routes, and -1 for invalid or duplicate selections.
    // - Produce outputs that are compatible with dispatch_impl when
    //   deterministic mode reuses dst_buffer_slot_idx.
    //
    // Placeholder: keep the header compilable while leaving all routes empty.
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto num_route_entries = num_tokens * kNumTopk;

    for (int rank_idx = thread_idx; rank_idx < kNumScaleupRanks; rank_idx += kNumThreads)
        rank_count_buffer[sm_idx * kNumScaleupRanks + rank_idx] = 0;

    for (int idx = sm_idx * kNumThreads + thread_idx; idx < num_route_entries; idx += kNumSMs * kNumThreads)
        dst_buffer_slot_idx[idx] = -1;
}

}  // namespace deep_ep::elastic
