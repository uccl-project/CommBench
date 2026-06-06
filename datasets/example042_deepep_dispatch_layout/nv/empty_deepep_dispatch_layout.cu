/*
 * DeepEP Dispatch Layout Benchmark (completion template)
 *
 * This file keeps the full test harness, CPU reference, benchmark, and JSON
 * output.  Only the DeepEP-like dispatch layout kernel body is removed.
 *
 * Complete the kernel so it computes:
 *
 *   1. num_tokens_per_expert
 *   2. num_tokens_per_rank
 *   3. is_token_in_rank
 *
 * Keep all class and function signatures unchanged.
 */

#include <cuda_runtime.h>

#include <algorithm>
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

using TopkIdx = int64_t;

// DeepEP assumes at most eight local NVLink peers per domain.  The RDMA path is
// removed here, but the constant is kept to mirror the original launch shape.
static constexpr int kNumMaxNvlPeers = 8;

// Rank deduplication uses a fixed-size per-token array in this dataset.
static constexpr int kMaxRanks = 8;

template <int kNumThreads, int kNumExpertsPerBlock, int kNumRanksPerBlock>
__global__ void deepep_get_dispatch_layout_kernel(
    const TopkIdx* topk_idx,
    int* num_tokens_per_rank,
    int* num_tokens_per_rdma_rank,
    int* num_tokens_per_expert,
    bool* is_token_in_rank,
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts) {
  // TODO:
  // Complete the dispatch layout outputs for topk_idx[num_tokens, num_topk].
  //
  //   1. num_tokens_per_expert[expert]
  //      Number of top-k selections for each expert.
  //
  //   2. is_token_in_rank[token, rank]
  //      Whether a token must be sent to a rank.
  //
  //   3. num_tokens_per_rank[rank]
  //      Number of tokens routed to each rank.  If one token selects multiple
  //      experts on the same rank, count that token/rank pair once.
  //
  // Constraints:
  //   - num_tokens_per_rdma_rank is always nullptr in this dataset.
  //   - num_experts is divisible by num_ranks.
  //   - num_topk <= 8, num_ranks <= 8.
  //   - Keep the kernel signature unchanged.
  (void)topk_idx;
  (void)num_tokens_per_rank;
  (void)num_tokens_per_rdma_rank;
  (void)num_tokens_per_expert;
  (void)is_token_in_rank;
  (void)num_tokens;
  (void)num_topk;
  (void)num_ranks;
  (void)num_experts;
}

class DeepepDispatchLayout {
 public:
  DeepepDispatchLayout(int num_tokens, int num_topk, int num_ranks,
                       int num_experts)
      : num_tokens_(num_tokens),
        num_topk_(num_topk),
        num_ranks_(num_ranks),
        num_experts_(num_experts) {
    if (num_ranks_ <= 0 || num_ranks_ > kMaxRanks) {
      std::fprintf(stderr, "num_ranks must be in [1, %d]\n", kMaxRanks);
      std::exit(EXIT_FAILURE);
    }
    if (num_topk_ <= 0 || num_topk_ > 8) {
      std::fprintf(stderr, "num_topk must be in [1, 8]\n");
      std::exit(EXIT_FAILURE);
    }
    if (num_experts_ % num_ranks_ != 0) {
      std::fprintf(stderr, "num_experts must be divisible by num_ranks\n");
      std::exit(EXIT_FAILURE);
    }
  }

  void compute(const TopkIdx* d_topk_idx, int* d_num_tokens_per_rank,
               int* d_num_tokens_per_expert, bool* d_is_token_in_rank,
               cudaStream_t stream = 0) const {
    constexpr int kNumThreads = 256;
    constexpr int kNumExpertsPerBlock = 4;
    constexpr int kNumRanksPerBlock = 8;
    static_assert(kNumRanksPerBlock % kNumMaxNvlPeers == 0,
                  "DeepEP layout assumes rank blocks align with NVL peers");

    int expert_blocks =
        (num_experts_ + kNumExpertsPerBlock - 1) / kNumExpertsPerBlock;
    int rank_blocks =
        (num_ranks_ + kNumRanksPerBlock - 1) / kNumRanksPerBlock;
    int num_blocks = expert_blocks + rank_blocks;

    deepep_get_dispatch_layout_kernel<kNumThreads, kNumExpertsPerBlock,
                                      kNumRanksPerBlock>
        <<<num_blocks, kNumThreads, 0, stream>>>(
            d_topk_idx, d_num_tokens_per_rank, nullptr,
            d_num_tokens_per_expert, d_is_token_in_rank, num_tokens_,
            num_topk_, num_ranks_, num_experts_);
    CUDA_CHECK(cudaGetLastError());
  }

 private:
  int num_tokens_ = 0;
  int num_topk_ = 0;
  int num_ranks_ = 0;
  int num_experts_ = 0;
};

struct LayoutCase {
  int num_tokens;
  int num_topk;
  int num_ranks;
  int num_experts;
  int pattern;
};

struct MetricRow {
  int data_size = 0;
  int num_topk = 0;
  int num_ranks = 0;
  int num_experts = 0;
  std::string pattern;
  double latency_avg_us = 0.0;
  double throughput_avg = 0.0;
};

enum RoutePattern {
  kRouteSpread = 0,
  kRouteRankSweep = 1,
  kRouteSameRank = 2,
};

static const char* routePatternName(int pattern) {
  switch (pattern) {
    case kRouteSpread:
      return "spread";
    case kRouteRankSweep:
      return "rank_sweep";
    case kRouteSameRank:
      return "same_rank";
    default:
      return "unknown";
  }
}

static std::vector<TopkIdx> makeTopk(const LayoutCase& c) {
  std::vector<TopkIdx> topk(static_cast<size_t>(c.num_tokens) * c.num_topk);
  int experts_per_rank = c.num_experts / c.num_ranks;
  for (int token = 0; token < c.num_tokens; ++token) {
    for (int k = 0; k < c.num_topk; ++k) {
      // Deterministic routing patterns cover different correctness and
      // benchmark stress points:
      //
      //   spread:     broadly distributes selections across experts.
      //   rank_sweep: makes each token touch multiple ranks when top-k allows.
      //   same_rank:  sends a token's top-k experts to one rank, checking that
      //               num_tokens_per_rank deduplicates token/rank pairs.
      int expert = 0;
      if (c.pattern == kRouteRankSweep) {
        int rank = (token + k) % c.num_ranks;
        int local_expert = (token * 5 + k * 3) % experts_per_rank;
        expert = rank * experts_per_rank + local_expert;
      } else if (c.pattern == kRouteSameRank) {
        int rank = (token * 3 + c.num_topk) % c.num_ranks;
        int local_expert = (token + k * 7) % experts_per_rank;
        expert = rank * experts_per_rank + local_expert;
      } else {
        expert = (token * 17 + k * 31 + c.num_ranks) % c.num_experts;
      }
      topk[static_cast<size_t>(token) * c.num_topk + k] = expert;
    }
  }
  return topk;
}

static void cpuReference(const LayoutCase& c, const std::vector<TopkIdx>& topk,
                         std::vector<int>* num_tokens_per_rank,
                         std::vector<int>* num_tokens_per_expert,
                         std::vector<unsigned char>* is_token_in_rank) {
  num_tokens_per_rank->assign(c.num_ranks, 0);
  num_tokens_per_expert->assign(c.num_experts, 0);
  is_token_in_rank->assign(static_cast<size_t>(c.num_tokens) * c.num_ranks, 0);

  int experts_per_rank = c.num_experts / c.num_ranks;
  for (int token = 0; token < c.num_tokens; ++token) {
    bool seen_rank[kMaxRanks] = {false};
    for (int k = 0; k < c.num_topk; ++k) {
      int expert =
          static_cast<int>(topk[static_cast<size_t>(token) * c.num_topk + k]);
      if (expert < 0) continue;
      ++(*num_tokens_per_expert)[expert];
      seen_rank[expert / experts_per_rank] = true;
    }
    for (int rank = 0; rank < c.num_ranks; ++rank) {
      if (seen_rank[rank]) {
        ++(*num_tokens_per_rank)[rank];
        (*is_token_in_rank)[static_cast<size_t>(token) * c.num_ranks + rank] =
            1;
      }
    }
  }
}

static bool runCorrectnessCase(const LayoutCase& c) {
  std::vector<TopkIdx> h_topk = makeTopk(c);
  std::vector<int> ref_rank;
  std::vector<int> ref_expert;
  std::vector<unsigned char> ref_flags;
  cpuReference(c, h_topk, &ref_rank, &ref_expert, &ref_flags);

  TopkIdx* d_topk = nullptr;
  int* d_rank = nullptr;
  int* d_expert = nullptr;
  bool* d_flags = nullptr;

  CUDA_CHECK(cudaMalloc(&d_topk, h_topk.size() * sizeof(TopkIdx)));
  CUDA_CHECK(cudaMalloc(&d_rank, c.num_ranks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_expert, c.num_experts * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_flags,
                        static_cast<size_t>(c.num_tokens) * c.num_ranks *
                            sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_rank, 0, c.num_ranks * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_expert, 0, c.num_experts * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_flags, 0,
                        static_cast<size_t>(c.num_tokens) * c.num_ranks *
                            sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(TopkIdx),
                        cudaMemcpyHostToDevice));

  DeepepDispatchLayout layout(c.num_tokens, c.num_topk, c.num_ranks,
                              c.num_experts);
  layout.compute(d_topk, d_rank, d_expert, d_flags);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> got_rank(c.num_ranks);
  std::vector<int> got_expert(c.num_experts);
  std::vector<unsigned char> got_flags(
      static_cast<size_t>(c.num_tokens) * c.num_ranks);

  CUDA_CHECK(cudaMemcpy(got_rank.data(), d_rank, got_rank.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_expert.data(), d_expert,
                        got_expert.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_flags.data(), d_flags, got_flags.size(),
                        cudaMemcpyDeviceToHost));

  bool ok = true;
  ok = ok && (got_rank == ref_rank);
  ok = ok && (got_expert == ref_expert);
  for (size_t i = 0; i < ref_flags.size(); ++i) {
    if ((got_flags[i] != 0) != (ref_flags[i] != 0)) {
      ok = false;
      break;
    }
  }

  CUDA_CHECK(cudaFree(d_topk));
  CUDA_CHECK(cudaFree(d_rank));
  CUDA_CHECK(cudaFree(d_expert));
  CUDA_CHECK(cudaFree(d_flags));
  return ok;
}

static bool runTest(const std::vector<LayoutCase>& correctness_cases) {
  bool pass = true;
  for (const LayoutCase& c : correctness_cases) {
    pass = runCorrectnessCase(c) && pass;
  }
  return pass;
}

static MetricRow benchmarkCase(const LayoutCase& c) {
  std::vector<TopkIdx> h_topk = makeTopk(c);

  TopkIdx* d_topk = nullptr;
  int* d_rank = nullptr;
  int* d_expert = nullptr;
  bool* d_flags = nullptr;

  CUDA_CHECK(cudaMalloc(&d_topk, h_topk.size() * sizeof(TopkIdx)));
  CUDA_CHECK(cudaMalloc(&d_rank, c.num_ranks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_expert, c.num_experts * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_flags,
                        static_cast<size_t>(c.num_tokens) * c.num_ranks *
                            sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(TopkIdx),
                        cudaMemcpyHostToDevice));

  DeepepDispatchLayout layout(c.num_tokens, c.num_topk, c.num_ranks,
                              c.num_experts);

  const int warmup_iters = 20;
  const int bench_iters = 100;
  for (int i = 0; i < warmup_iters; ++i) {
    layout.compute(d_topk, d_rank, d_expert, d_flags);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; ++i) {
    layout.compute(d_topk, d_rank, d_expert, d_flags);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  double avg_us = static_cast<double>(elapsed_ms) * 1000.0 / bench_iters;
  double mtokens_per_sec =
      avg_us > 0.0 ? static_cast<double>(c.num_tokens) / avg_us : 0.0;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_topk));
  CUDA_CHECK(cudaFree(d_rank));
  CUDA_CHECK(cudaFree(d_expert));
  CUDA_CHECK(cudaFree(d_flags));

  MetricRow row;
  row.data_size = c.num_tokens;
  row.num_topk = c.num_topk;
  row.num_ranks = c.num_ranks;
  row.num_experts = c.num_experts;
  row.pattern = routePatternName(c.pattern);
  row.latency_avg_us = avg_us;
  row.throughput_avg = mtokens_per_sec;
  return row;
}

static void printJson(bool pass, const std::vector<MetricRow>& metrics) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "{\n";
  std::cout << "  \"Correctness\": \"" << (pass ? "PASS" : "FAIL")
            << "\",\n";
  std::cout << "  \"data_size_unit\": \"tokens\",\n";
  std::cout << "  \"throughput_unit\": \"Mtokens/s\",\n";
  std::cout << "  \"latency_unit\": \"us\",\n";
  std::cout << "  \"metrics\": [\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    const MetricRow& r = metrics[i];
    std::cout << "    {\"data_size\": " << r.data_size
              << ", \"num_topk\": " << r.num_topk
              << ", \"num_ranks\": " << r.num_ranks
              << ", \"num_experts\": " << r.num_experts
              << ", \"pattern\": \"" << r.pattern << "\""
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
  if (err != cudaSuccess || device_count <= 0) {
    std::cout << "{\"Correctness\": \"SKIP\", "
              << "\"reason\": \"no CUDA device available\"}\n";
    return 0;
  }
  CUDA_CHECK(cudaSetDevice(0));

  std::vector<LayoutCase> correctness_cases;
  correctness_cases.push_back({257, 1, 1, 4, kRouteSpread});
  for (int ranks : {2, 4, 8}) {
    correctness_cases.push_back({257, 1, ranks, ranks * 4, kRouteSpread});
    correctness_cases.push_back({1024, 2, ranks, ranks * 4, kRouteSpread});
    correctness_cases.push_back({4096, 4, ranks, ranks * 4,
                                 kRouteRankSweep});
    correctness_cases.push_back({4096, 8, ranks, ranks * 4,
                                 kRouteSameRank});
  }

  bool pass = runTest(correctness_cases);

  std::vector<MetricRow> metrics;
  // Benchmark only times the layout kernel itself: input generation, H2D copy,
  // allocation, and correctness reads are outside the CUDA event interval.
  // Token scaling stays comparable with earlier datasets, while the extra
  // top-k/pattern rows exercise the expensive rank-flag and dedup paths.
  for (int tokens : {1024, 4096, 16384, 65536}) {
    metrics.push_back(benchmarkCase({tokens, 2, 8, 32, kRouteSpread}));
  }
  metrics.push_back(benchmarkCase({65536, 4, 8, 32, kRouteRankSweep}));
  metrics.push_back(benchmarkCase({65536, 8, 8, 32, kRouteSameRank}));
  metrics.push_back(benchmarkCase({65536, 2, 4, 32, kRouteSpread}));

  printJson(pass, metrics);
  return pass ? 0 : 1;
}
