// Intra-CTA producer/consumer synchronization using a Hopper/Blackwell
// shared-memory mbarrier with a non-blocking try_wait probe.
//
// AI completion task: implement the kernel logic and the four `kernel::*`
// PTX helpers marked with `// TODO`.  The driver, JSON output, sweep
// (blockDim.x = 32 / 64 / 128 / 256), and timing scaffolding are
// intentionally left intact.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,    \
                   __LINE__, cudaGetErrorString(_e));                          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#if !defined(MIRAGE_GRACE_HOPPER) && !defined(MIRAGE_GRACE_BLACKWELL)
#define MIRAGE_GRACE_HOPPER
#endif

namespace kernel {

struct Barrier {
 private:
  uint64_t value;

 public:
  __device__ inline uint64_t get_value() { return value; }
};

__device__ static inline void initialize_barrier(Barrier& smem_barrier,
                                                 int thread_count = 1) {
  // TODO: emit `mbarrier.init.shared::cta.b64` for `smem_barrier`,
  // initialised to expect `thread_count` arrives.
}

__device__ static inline void arrive(Barrier& smem_barrier,
                                     uint32_t count = 1) {
  // TODO: emit `mbarrier.arrive.shared::cta.b64` adding `count` arrives.
}

// Non-blocking probe — must match mirage's try_wait_barrier semantics.
__device__ static inline bool try_wait_barrier(uint64_t& smem_barrier,
                                               uint32_t phase) {
  // TODO: emit `mbarrier.try_wait.parity.shared::cta.b64` and return whether
  // the barrier has flipped to the requested phase.
  return false;
}

}  // namespace kernel

constexpr int kMaxBlockThreads = 256;
constexpr int kMaxProducers = kMaxBlockThreads - 1;
constexpr int kProbeBound = 1 << 20;

__global__ void try_wait_bench_kernel(int* __restrict__ out,
                                      unsigned long long* __restrict__ probes,
                                      int iters) {
  __shared__ uint64_t mbar;
  __shared__ int tile[kMaxProducers];

  // TODO: implement the per-iteration protocol with `producers = blockDim.x - 1`:
  //   1. thread 0 re-initialises `mbar` to expect `producers` arrivers
  //   2. each producer (tid >= 1) writes tile[tid-1] = it * producers + (tid-1)
  //      then arrives on the barrier with count = 1
  //   3. thread 0 (consumer) polls `try_wait_barrier(mbar, 0)` until it
  //      returns true, accumulating per-iteration probe counts
  //   4. consumer copies `producers` ints out to
  //      `out[it * kMaxProducers + slot]` for slot in [0, producers)
  //   5. write the total probe count into *probes from the consumer
}

class BarrierBench {
 public:
  BarrierBench() {
    CUDA_CHECK(cudaMalloc(&d_probes_, sizeof(unsigned long long)));
  }

  ~BarrierBench() {
    if (d_out_) {
      cudaFree(d_out_);
    }
    if (d_probes_) {
      cudaFree(d_probes_);
    }
  }

  void runOnce(int block_threads, int iters, cudaStream_t stream = nullptr) {
    ensureCapacity(iters);
    try_wait_bench_kernel<<<1, block_threads, 0, stream>>>(d_out_, d_probes_,
                                                            iters);
  }

  bool checkCorrectness(int block_threads, int iters) {
    const int producers = block_threads - 1;
    CUDA_CHECK(cudaMemset(d_probes_, 0, sizeof(unsigned long long)));
    runOnce(block_threads, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> host(static_cast<size_t>(iters) * kMaxProducers);
    CUDA_CHECK(cudaMemcpy(host.data(), d_out_, host.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int it = 0; it < iters; ++it) {
      for (int slot = 0; slot < producers; ++slot) {
        int got = host[it * kMaxProducers + slot];
        int want = it * producers + slot;
        if (got != want) {
          std::fprintf(stderr,
                       "[BarrierBench] mismatch block=%d iter=%d slot=%d "
                       "got=%d want=%d\n",
                       block_threads, it, slot, got, want);
          return false;
        }
      }
    }
    unsigned long long probes = 0;
    CUDA_CHECK(cudaMemcpy(&probes, d_probes_, sizeof(probes),
                          cudaMemcpyDeviceToHost));
    if (probes < static_cast<unsigned long long>(iters)) {
      std::fprintf(stderr,
                   "[BarrierBench] probe count %llu < iters %d (try_wait never polled?)\n",
                   probes, iters);
      return false;
    }
    return true;
  }

  struct Sample {
    double latency_us_per_iter;
    double throughput_gbps;
    double avg_probes_per_iter;
    int block_threads;
    int iters;
    size_t bytes;
  };

  Sample benchmark(int block_threads, int iters, int repeats = 5,
                   int warmups = 2) {
    ensureCapacity(iters);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int w = 0; w < warmups; ++w) {
      runOnce(block_threads, iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const int producers = block_threads - 1;
    Sample best;
    best.latency_us_per_iter = 1e30;
    best.throughput_gbps = 0.0;
    best.avg_probes_per_iter = 0.0;
    best.block_threads = block_threads;
    best.iters = iters;
    best.bytes = static_cast<size_t>(iters) * producers * sizeof(int);

    for (int r = 0; r < repeats; ++r) {
      CUDA_CHECK(cudaMemset(d_probes_, 0, sizeof(unsigned long long)));
      CUDA_CHECK(cudaEventRecord(start));
      runOnce(block_threads, iters);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      double per_iter_us =
          static_cast<double>(ms) * 1000.0 / static_cast<double>(iters);
      double bits = static_cast<double>(best.bytes) * 8.0;
      double gbps = bits / (static_cast<double>(ms) / 1000.0) / 1e9;

      unsigned long long probes = 0;
      CUDA_CHECK(cudaMemcpy(&probes, d_probes_, sizeof(probes),
                            cudaMemcpyDeviceToHost));
      double avg_probes =
          static_cast<double>(probes) / static_cast<double>(iters);

      if (per_iter_us < best.latency_us_per_iter) {
        best.latency_us_per_iter = per_iter_us;
        best.throughput_gbps = gbps;
        best.avg_probes_per_iter = avg_probes;
      }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
  }

 private:
  void ensureCapacity(int iters) {
    size_t needed = static_cast<size_t>(iters) * kMaxProducers * sizeof(int);
    if (needed <= cap_bytes_) {
      return;
    }
    if (d_out_) {
      CUDA_CHECK(cudaFree(d_out_));
    }
    CUDA_CHECK(cudaMalloc(&d_out_, needed));
    cap_bytes_ = needed;
  }

  int* d_out_ = nullptr;
  unsigned long long* d_probes_ = nullptr;
  size_t cap_bytes_ = 0;
};

static void runTest() {
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  if (prop.major < 9) {
    std::fprintf(stderr,
                 "ERROR: this example requires Hopper (sm_90+); detected sm_%d%d (%s)\n",
                 prop.major, prop.minor, prop.name);
    std::printf("{\"Correctness\": \"FAIL\"}\n");
    std::exit(1);
  }

  BarrierBench bench;

  const std::vector<int> block_threads_sweep = {32, 64, 128, 256};
  const int kBenchIters = 4096;
  const int kCorrIters = 256;

  bool ok = true;
  for (int bt : block_threads_sweep) {
    if (!bench.checkCorrectness(bt, kCorrIters)) {
      ok = false;
      break;
    }
  }

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", ok ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"threads\",\n");
  std::printf("  \"throughput_unit\": \"Gbps\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"avg_probes_unit\": \"count\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < block_threads_sweep.size(); ++i) {
    int bt = block_threads_sweep[i];
    auto s = bench.benchmark(bt, kBenchIters);
    std::printf(
        "    {\"data_size\": %d, \"latency_avg\": %.4f, "
        "\"throughput_avg\": %.4f, \"avg_probes\": %.3f, "
        "\"producers\": %d, \"iters\": %d}%s\n",
        bt, s.latency_us_per_iter, s.throughput_gbps, s.avg_probes_per_iter,
        bt - 1, s.iters, (i + 1 < block_threads_sweep.size()) ? "," : "");
  }
  std::printf("  ]\n");
  std::printf("}\n");

  if (!ok) {
    std::exit(1);
  }
}

int main() {
  runTest();
  return 0;
}
