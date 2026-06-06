#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <bool kIsScaleupNVLink,
          bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumDispatchWarps,
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumDispatchThreads = kNumDispatchWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumDispatchThreads,
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* buffer,
    void* workspace, void* mapped_host_workspace,
    const int rank_idx
) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // Gin handle
    // We treat each warp as a "channel"
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(
        sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Barrier without TMA store flush, without prologue grid sync
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // TODO: Implement DeepEP's notify/count role. The real path counts
        // per-destination scale-up ranks and local experts in shared memory,
        // performs the full-grid reduction, exchanges rank/expert counts with
        // peers using system-scope release/acquire semantics, writes CPU-sync
        // counters, and produces the scale-up/expert prefix sums consumed by
        // the copy epilogue.
        //
        // This placeholder publishes ready zero counters so the harness fails
        // by correctness instead of timing out while waiting for CPU-sync
        // metadata.
        if (sm_idx == 0) {
            if (warp_idx == 0) {
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                    psum_num_recv_tokens_per_scaleup_rank[i] = 0;
                    if constexpr (kDoCPUSync)
                        host_workspace_layout.get_scaleup_rank_count_ptr<false>()[i] = math::encode_decode_positive(0);
                }
            }
            if (warp_idx == 1) {
                for (int i = lane_idx; i < kNumExpertsPerRank + 1; i += 32)
                    psum_num_recv_tokens_per_expert[i] = 0;
                if constexpr (kDoCPUSync) {
                    for (int i = lane_idx; i < kNumExpertsPerRank; i += 32)
                        host_workspace_layout.get_scaleup_expert_count_ptr<false>()[i] = math::encode_decode_positive(0);
                }
            }
        }
    } else {
        const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;

        // Buffer layouts
        const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
        const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumDispatchWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(dispatch_warp_idx).get_token_buffer(0);
        auto recv_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
        auto send_buffer = layout::BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());
        recv_buffer = recv_buffer.get_rank_buffer(rank_idx);

        // Init TMA
        ptx::arrival_phase phase = 0;
        const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
        if (ptx::elect_one_sync())
            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        __syncwarp();

        // Iterate all tokens
        // TODO: Implementing the direct scale-up dispatch payload role belongs
        // to example80. This notify-only template intentionally skips payload
        // movement so the notify/count path remains the substantive hole here.
        const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
        const auto token_stride = kNumDispatchWarps * kNumSMs;
        for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
            if constexpr (kReuseSlotIndices) {
                __syncwarp();
            } else {
                if (lane_idx < kNumTopk)
                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = -1;
                __syncwarp();
            }
        }
    }

    // Barrier to ensure data arrival
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag1, true, true, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Clean atomic counters
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::elastic
