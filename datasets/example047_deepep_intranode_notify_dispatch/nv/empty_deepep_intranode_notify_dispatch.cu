// DeepEP intra-node notify_dispatch standalone CUDA dataset (completion
// template).
//
// Source kernel:
//
//   DeepEP/csrc/kernels/intranode.cu
//     deep_ep::intranode::notify_dispatch(...)
//
// Source helpers:
//
//   DeepEP/csrc/kernels/utils.cuh
//     deep_ep::barrier_block(...)
//     deep_ep::get_channel_task_range(...)
//     deep_ep::warp_reduce_sum(...)
//
// This dataset removes only the parts that are outside the standalone notify
// metadata task:
//
//   - moe_recv_counter_mapped is omitted.  Correctness reads
//     rank_prefix_matrix_copy directly.
//   - num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts, and
//     expert_alignment are omitted.  Per-expert receive counters belong to
//     allocation/accounting, while this dataset targets dispatch metadata.
//   - num_memset_int queue-workspace clearing is omitted.  The later queue path
//     is covered by a separate dataset and owns its own reset logic.
//   - buffer_ptrs is represented as int** rather than DeepEP's void** because
//     the standalone buffer contains only the rank count/prefix matrix.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(_e));                                  \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

[[maybe_unused]] static constexpr int FINISHED_SUM_TAG = 1024;

struct MetricRow {
  int data_size = 0;
  double latency_avg_us = 0.0;
  double throughput_avg = 0.0;
};

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
  return *((volatile const int*)ptr);
}

__device__ __forceinline__ int warp_reduce_sum(int value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void get_channel_task_range(
    int num_tokens, int num_channels, int channel_id, int& token_start_idx,
    int& token_end_idx) {
  int tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
  token_start_idx = min(tokens_per_channel * channel_id, num_tokens);
  token_end_idx = min(token_start_idx + tokens_per_channel, num_tokens);
}

template <int kNumRanks, bool kSyncOnly = false>
__device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
  auto thread_id = static_cast<int>(threadIdx.x);

  if constexpr (!kSyncOnly) {
    // DeepEP calls memory_fence() which emits `fence.acq_rel.sys` (PTX
    // acquire-release fence).  __threadfence_system() emits `membar.sys` which
    // provides the same system-scope ordering guarantee needed before the
    // cross-GPU atomics.  utils.cuh is not included in this standalone file.
    __threadfence_system();
    __syncthreads();
  }

  if (thread_id < kNumRanks) {
    atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
    atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
  }

  while (true) {
    int done = 1;
    if (thread_id < kNumRanks) {
      done = (ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) <= 0);
    }
    unsigned mask = __ballot_sync(0xffffffffu, done);
    unsigned want = (1u << kNumRanks) - 1u;
    if ((mask & want) == want) break;
  }

  __syncthreads();
}

template <int kNumRanks>
__global__ void notify_dispatch(
    const int* num_tokens_per_rank,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy,
    int** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_tokens,
    int num_channels) {
  // TODO:
  //
  // Complete notify_dispatch so it produces the two metadata outputs checked by
  // the harness:
  //
  //   1. rank_prefix_matrix_copy[src_rank, dst_rank]
  //      For the local rank, this stores the cumulative number of tokens sent
  //      from source ranks [0, src_rank] to dst_rank.
  //
  //   2. channel_prefix_matrix[dst_rank, channel_id]
  //      For this source rank, this stores the cumulative number of selected
  //      tokens for dst_rank from channel 0 through channel_id.
  //
  // Required behavior:
  //   - blockIdx.x == 0 handles the cross-rank count exchange and writes
  //     rank_prefix_matrix_copy.
  //   - blockIdx.x in [1, kNumRanks] handles channel_prefix_matrix for
  //     dst_rank = blockIdx.x - 1.
  //   - buffer_ptrs[dst] points to dst rank's kNumRanks x kNumRanks count
  //     matrix.  Store num_tokens_per_rank[dst] at [rank, dst].
  //   - is_token_in_rank is laid out as [num_tokens, kNumRanks].
  //
  // Constraints:
  //   - Keep this kernel signature unchanged.
  //   - kNumRanks is one of 2, 4, or 8 in the test harness.
  //   - num_channels is positive.
  //   - The helper functions above may be used.
  (void)num_tokens_per_rank;
  (void)is_token_in_rank;
  (void)channel_prefix_matrix;
  (void)rank_prefix_matrix_copy;
  (void)buffer_ptrs;
  (void)barrier_signal_ptrs;
  (void)rank;
  (void)num_tokens;
  (void)num_channels;
}

static void getChannelTaskRangeHost(int num_tokens, int num_channels,
                                    int channel_id, int* token_start,
                                    int* token_end) {
  int tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
  *token_start = std::min(tokens_per_channel * channel_id, num_tokens);
  *token_end = std::min(*token_start + tokens_per_channel, num_tokens);
}

static bool hasFullPeerAccess(int num_ranks) {
  for (int i = 0; i < num_ranks; ++i) {
    for (int j = 0; j < num_ranks; ++j) {
      if (i == j) continue;
      int can_access = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
      if (!can_access) return false;
    }
  }
  return true;
}

static std::vector<int> candidateRanks(int device_count) {
  std::vector<int> out;
  for (int n : {2, 4, 8}) {
    if (n <= device_count && hasFullPeerAccess(n)) out.push_back(n);
  }
  return out;
}

static int routeDst(int rank, int token, int slot, int num_ranks) {
  if (num_ranks == 1) return 0;
  int stride = (rank % (num_ranks - 1)) + 1;
  return (token + rank + slot * stride) % num_ranks;
}

class IntranodeNotifyDispatch {
 public:
  IntranodeNotifyDispatch() = default;
  ~IntranodeNotifyDispatch() { teardown(); }

  void setup(int num_ranks, int num_tokens, int num_channels) {
    teardown();
    num_ranks_ = num_ranks;
    num_tokens_ = num_tokens;
    num_channels_ = num_channels;
    enablePeerAccess();

    num_tokens_per_rank_.resize(num_ranks_);
    is_token_in_rank_.resize(num_ranks_);
    buffer_storage_.resize(num_ranks_);
    rank_prefix_matrix_copy_.resize(num_ranks_);
    channel_prefix_matrix_.resize(num_ranks_);
    barrier_signal_storage_.resize(num_ranks_);
    buffer_ptrs_.resize(num_ranks_);
    barrier_signal_ptrs_.resize(num_ranks_);
    host_flags_.assign(num_ranks_, {});
    host_counts_.assign(num_ranks_ * num_ranks_, 0);

    for (int rank = 0; rank < num_ranks_; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(
          cudaMalloc(&num_tokens_per_rank_[rank], num_ranks_ * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&is_token_in_rank_[rank],
                            static_cast<size_t>(num_tokens_) * num_ranks_ *
                                sizeof(bool)));
      CUDA_CHECK(cudaMalloc(&buffer_storage_[rank],
                            num_ranks_ * num_ranks_ * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&rank_prefix_matrix_copy_[rank],
                            num_ranks_ * num_ranks_ * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&channel_prefix_matrix_[rank],
                            num_ranks_ * num_channels_ * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&barrier_signal_storage_[rank],
                            num_ranks_ * sizeof(int)));
    }

    for (int rank = 0; rank < num_ranks_; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMalloc(&buffer_ptrs_[rank],
                            num_ranks_ * sizeof(int*)));
      CUDA_CHECK(cudaMemcpy(buffer_ptrs_[rank], buffer_storage_.data(),
                            num_ranks_ * sizeof(int*),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&barrier_signal_ptrs_[rank],
                            num_ranks_ * sizeof(int*)));
      CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_[rank],
                            barrier_signal_storage_.data(),
                            num_ranks_ * sizeof(int*),
                            cudaMemcpyHostToDevice));
    }

    makeInputs();
    reset();
  }

  void teardown() {
    for (int rank = 0; rank < num_ranks_; ++rank) {
      cudaSetDevice(rank);
      if (rank < static_cast<int>(num_tokens_per_rank_.size()) &&
          num_tokens_per_rank_[rank])
        cudaFree(num_tokens_per_rank_[rank]);
      if (rank < static_cast<int>(is_token_in_rank_.size()) &&
          is_token_in_rank_[rank])
        cudaFree(is_token_in_rank_[rank]);
      if (rank < static_cast<int>(buffer_storage_.size()) && buffer_storage_[rank])
        cudaFree(buffer_storage_[rank]);
      if (rank < static_cast<int>(rank_prefix_matrix_copy_.size()) &&
          rank_prefix_matrix_copy_[rank])
        cudaFree(rank_prefix_matrix_copy_[rank]);
      if (rank < static_cast<int>(channel_prefix_matrix_.size()) &&
          channel_prefix_matrix_[rank])
        cudaFree(channel_prefix_matrix_[rank]);
      if (rank < static_cast<int>(barrier_signal_storage_.size()) &&
          barrier_signal_storage_[rank])
        cudaFree(barrier_signal_storage_[rank]);
      if (rank < static_cast<int>(buffer_ptrs_.size()) &&
          buffer_ptrs_[rank])
        cudaFree(buffer_ptrs_[rank]);
      if (rank < static_cast<int>(barrier_signal_ptrs_.size()) &&
          barrier_signal_ptrs_[rank])
        cudaFree(barrier_signal_ptrs_[rank]);
    }
    disablePeerAccess();
    num_tokens_per_rank_.clear();
    is_token_in_rank_.clear();
    buffer_storage_.clear();
    rank_prefix_matrix_copy_.clear();
    channel_prefix_matrix_.clear();
    barrier_signal_storage_.clear();
    buffer_ptrs_.clear();
    barrier_signal_ptrs_.clear();
    host_flags_.clear();
    host_counts_.clear();
    num_ranks_ = 0;
    num_tokens_ = 0;
    num_channels_ = 0;
  }

  void reset() {
    for (int rank = 0; rank < num_ranks_; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMemset(buffer_storage_[rank], 0,
                            num_ranks_ * num_ranks_ * sizeof(int)));
      CUDA_CHECK(cudaMemset(rank_prefix_matrix_copy_[rank], 0,
                            num_ranks_ * num_ranks_ * sizeof(int)));
      CUDA_CHECK(cudaMemset(channel_prefix_matrix_[rank], 0,
                            num_ranks_ * num_channels_ * sizeof(int)));
      CUDA_CHECK(cudaMemset(barrier_signal_storage_[rank], 0,
                            num_ranks_ * sizeof(int)));
    }
    syncAll();
  }

  void runOnce(bool reset_before_run = true) {
    if (reset_before_run) reset();
    constexpr int kThreads = 128;
    dim3 grid(1 + num_ranks_);
    for (int rank = 0; rank < num_ranks_; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      switch (num_ranks_) {
        case 2:
          notify_dispatch<2><<<grid, kThreads>>>(
              num_tokens_per_rank_[rank], is_token_in_rank_[rank],
              channel_prefix_matrix_[rank], rank_prefix_matrix_copy_[rank],
              buffer_ptrs_[rank], barrier_signal_ptrs_[rank], rank,
              num_tokens_, num_channels_);
          break;
        case 4:
          notify_dispatch<4><<<grid, kThreads>>>(
              num_tokens_per_rank_[rank], is_token_in_rank_[rank],
              channel_prefix_matrix_[rank], rank_prefix_matrix_copy_[rank],
              buffer_ptrs_[rank], barrier_signal_ptrs_[rank], rank,
              num_tokens_, num_channels_);
          break;
        case 8:
          notify_dispatch<8><<<grid, kThreads>>>(
              num_tokens_per_rank_[rank], is_token_in_rank_[rank],
              channel_prefix_matrix_[rank], rank_prefix_matrix_copy_[rank],
              buffer_ptrs_[rank], barrier_signal_ptrs_[rank], rank,
              num_tokens_, num_channels_);
          break;
        default:
          std::fprintf(stderr, "unsupported num_ranks: %d\n", num_ranks_);
          std::exit(EXIT_FAILURE);
      }
      CUDA_CHECK(cudaGetLastError());
    }
    syncAll();
  }

  bool check() {
    bool ok = true;

    // --- rank_prefix_matrix_copy check ---
    // Each GPU dst owns column dst of the prefix matrix.
    // rank_prefix_matrix_copy[src * num_ranks + dst] = cumulative tokens
    // from ranks 0..src going to rank dst.
    std::vector<int> expected_rank_prefix(num_ranks_ * num_ranks_, 0);
    for (int dst = 0; dst < num_ranks_; ++dst) {
      int running = 0;
      for (int src = 0; src < num_ranks_; ++src) {
        running += host_counts_[src * num_ranks_ + dst];
        expected_rank_prefix[src * num_ranks_ + dst] = running;
      }
    }

    for (int dst = 0; dst < num_ranks_; ++dst) {
      std::vector<int> got(num_ranks_ * num_ranks_);
      CUDA_CHECK(cudaSetDevice(dst));
      CUDA_CHECK(cudaMemcpy(got.data(), rank_prefix_matrix_copy_[dst],
                            got.size() * sizeof(int),
                            cudaMemcpyDeviceToHost));
      for (int src = 0; src < num_ranks_; ++src) {
        int idx = src * num_ranks_ + dst;
        if (got[idx] != expected_rank_prefix[idx]) {
          if (ok) {
            std::fprintf(stderr,
                "rank_prefix_matrix_copy FAIL: src=%d dst=%d "
                "expected=%d got=%d (num_tokens=%d num_channels=%d)\n",
                src, dst, expected_rank_prefix[idx], got[idx],
                num_tokens_, num_channels_);
          }
          ok = false;
        }
      }
    }

    // --- channel_prefix_matrix check ---
    // Each GPU src owns its own channel_prefix_matrix.
    // channel_prefix_matrix[dst * num_channels + ch] = cumulative tokens
    // that src sends to dst across channels 0..ch.
    for (int src = 0; src < num_ranks_; ++src) {
      std::vector<int> expected(num_ranks_ * num_channels_, 0);
      for (int dst = 0; dst < num_ranks_; ++dst) {
        int running = 0;
        for (int channel = 0; channel < num_channels_; ++channel) {
          int token_start = 0;
          int token_end = 0;
          getChannelTaskRangeHost(num_tokens_, num_channels_, channel,
                                  &token_start, &token_end);
          int count = 0;
          for (int token = token_start; token < token_end; ++token) {
            count += host_flags_[src][token * num_ranks_ + dst] != 0;
          }
          running += count;
          expected[dst * num_channels_ + channel] = running;
        }
      }

      std::vector<int> got(num_ranks_ * num_channels_);
      CUDA_CHECK(cudaSetDevice(src));
      CUDA_CHECK(cudaMemcpy(got.data(), channel_prefix_matrix_[src],
                            got.size() * sizeof(int),
                            cudaMemcpyDeviceToHost));
      for (int dst = 0; dst < num_ranks_; ++dst) {
        for (int channel = 0; channel < num_channels_; ++channel) {
          int idx = dst * num_channels_ + channel;
          if (got[idx] != expected[idx]) {
            if (ok) {
              std::fprintf(stderr,
                  "channel_prefix_matrix FAIL: src=%d dst=%d ch=%d "
                  "expected=%d got=%d (num_tokens=%d num_channels=%d)\n",
                  src, dst, channel, expected[idx], got[idx],
                  num_tokens_, num_channels_);
            }
            ok = false;
          }
        }
      }
    }
    return ok;
  }

  int totalRoutes() const {
    return std::accumulate(host_counts_.begin(), host_counts_.end(), 0);
  }

 private:
  void makeInputs() {
    int num_topk = std::min(2, num_ranks_);
    for (int rank = 0; rank < num_ranks_; ++rank) {
      std::vector<uint8_t> flags(static_cast<size_t>(num_tokens_) * num_ranks_,
                                 0);
      for (int token = 0; token < num_tokens_; ++token) {
        for (int slot = 0; slot < num_topk; ++slot) {
          int dst = routeDst(rank, token, slot, num_ranks_);
          flags[token * num_ranks_ + dst] = 1;
        }
      }

      std::vector<int> num_tokens_per_rank(num_ranks_, 0);
      for (int token = 0; token < num_tokens_; ++token) {
        for (int dst = 0; dst < num_ranks_; ++dst) {
          num_tokens_per_rank[dst] += flags[token * num_ranks_ + dst] != 0;
        }
      }

      for (int dst = 0; dst < num_ranks_; ++dst) {
        host_counts_[rank * num_ranks_ + dst] = num_tokens_per_rank[dst];
      }
      host_flags_[rank] = flags;

      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMemcpy(is_token_in_rank_[rank], flags.data(),
                            flags.size() * sizeof(bool),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(num_tokens_per_rank_[rank],
                            num_tokens_per_rank.data(),
                            num_tokens_per_rank.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
    }
  }

  void enablePeerAccess() {
    for (int i = 0; i < num_ranks_; ++i) {
      CUDA_CHECK(cudaSetDevice(i));
      for (int j = 0; j < num_ranks_; ++j) {
        if (i == j) continue;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
        if (can_access) {
          cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
          if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            CUDA_CHECK(err);
          }
          cudaGetLastError();
        }
      }
    }
  }

  void disablePeerAccess() {
    for (int i = 0; i < num_ranks_; ++i) {
      cudaSetDevice(i);
      for (int j = 0; j < num_ranks_; ++j) {
        if (i == j) continue;
        cudaError_t err = cudaDeviceDisablePeerAccess(j);
        if (err != cudaSuccess && err != cudaErrorPeerAccessNotEnabled) {
          CUDA_CHECK(err);
        }
        cudaGetLastError();
      }
    }
  }

  void syncAll() {
    for (int rank = 0; rank < num_ranks_; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  int num_ranks_ = 0;
  int num_tokens_ = 0;
  int num_channels_ = 0;
  std::vector<int*> num_tokens_per_rank_;
  std::vector<bool*> is_token_in_rank_;
  std::vector<int*> buffer_storage_;
  std::vector<int*> rank_prefix_matrix_copy_;
  std::vector<int*> channel_prefix_matrix_;
  std::vector<int*> barrier_signal_storage_;
  std::vector<int**> buffer_ptrs_;
  std::vector<int**> barrier_signal_ptrs_;
  std::vector<std::vector<uint8_t>> host_flags_;
  std::vector<int> host_counts_;
};

static bool checkCorrectness(int num_ranks, int num_tokens, int num_channels) {
  IntranodeNotifyDispatch notify;
  notify.setup(num_ranks, num_tokens, num_channels);
  notify.runOnce(true);
  return notify.check();
}

static MetricRow benchmark(int num_ranks, int num_tokens, int num_channels) {
  IntranodeNotifyDispatch notify;
  notify.setup(num_ranks, num_tokens, num_channels);

  const int warmup_iters = 20;
  const int bench_iters = 120;
  for (int i = 0; i < warmup_iters; ++i) notify.runOnce(false);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < bench_iters; ++i) notify.runOnce(false);
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  double avg_us = total_us / bench_iters;
  MetricRow row;
  row.data_size = notify.totalRoutes();
  row.latency_avg_us = avg_us;
  row.throughput_avg = avg_us > 0.0 ? notify.totalRoutes() / avg_us : 0.0;
  return row;
}

static void printSkip(const char* reason) {
  std::cout << "{\n";
  std::cout << "  \"Correctness\": \"SKIP\",\n";
  std::cout << "  \"reason\": \"" << reason << "\"\n";
  std::cout << "}\n";
}

static void printJson(bool pass, const std::vector<MetricRow>& metrics) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "{\n";
  std::cout << "  \"Correctness\": \"" << (pass ? "PASS" : "FAIL")
            << "\",\n";
  std::cout << "  \"data_size_unit\": \"token-routes\",\n";
  std::cout << "  \"throughput_unit\": \"Mtoken-routes/s\",\n";
  std::cout << "  \"latency_unit\": \"us\",\n";
  std::cout << "  \"metrics\": [\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    const MetricRow& r = metrics[i];
    std::cout << "    {\"data_size\": " << r.data_size
              << ", \"throughput_avg\": " << r.throughput_avg
              << ", \"latency_avg\": " << r.latency_avg_us << "}";
    if (i + 1 != metrics.size()) std::cout << ",";
    std::cout << "\n";
  }
  std::cout << "  ]\n";
  std::cout << "}\n";
}

int main() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count < 2) {
    printSkip("need at least 2 CUDA devices");
    return 0;
  }

  std::vector<int> ranks_to_test = candidateRanks(std::min(device_count, 8));
  if (ranks_to_test.empty()) {
    printSkip("need at least 2 peer-accessible NVIDIA GPUs");
    return 0;
  }

  bool pass = true;
  std::vector<MetricRow> metrics;
  for (int num_ranks : ranks_to_test) {
    int num_channels = std::min(16, std::max(4, num_ranks * 2));
    // Three token counts exercise different get_channel_task_range paths:
    //   7     - small; many channels empty; last channel fewer tokens
    //   4096  - typical production size; divides cleanly by common num_channels
    //   4097  - one extra token; last channel gets fewer than tokens_per_channel
    for (int num_tokens : {7, 4096, 4097}) {
      pass = checkCorrectness(num_ranks, num_tokens, num_channels) && pass;
    }
    for (int num_tokens : {1024, 8192, 32768}) {
      metrics.push_back(benchmark(num_ranks, num_tokens, num_channels));
    }
  }

  printJson(pass, metrics);
  return pass ? 0 : 1;
}
