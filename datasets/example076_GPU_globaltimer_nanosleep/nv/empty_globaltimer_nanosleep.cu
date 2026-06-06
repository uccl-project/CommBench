// GPU %globaltimer + __nanosleep accuracy probe.
//
// AI completion task: implement `globaltimer_ns()` and the
// `nanosleep_probe_kernel` body marked `// TODO`.  Buffer setup, sweep,
// timing, correctness check, and JSON output are left intact.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

__device__ __forceinline__ uint64_t globaltimer_ns() {
  // TODO: read the GPU's nanosecond global timer and return it.  The
  // canonical PTX is `mov.u64 %r, %globaltimer;` (same source as
  // cuda::std::chrono::high_resolution_clock).
  return 0;
}

__global__ void nanosleep_probe_kernel(uint64_t* ts, int requested_ns) {
  // TODO: thread 0 of each block records globaltimer_ns() into
  //   ts[blockIdx.x], every thread calls __nanosleep(requested_ns), then
  //   thread 0 records globaltimer_ns() into ts[gridDim.x + blockIdx.x].
}

class SleepBench {
 public:
  SleepBench(int dev, int num_blocks)
      : dev_(dev), num_blocks_(num_blocks) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaMalloc(&d_ts_, 2 * num_blocks_ * sizeof(uint64_t)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }

  ~SleepBench() {
    if (d_ts_) {
      cudaSetDevice(dev_);
      cudaFree(d_ts_);
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  struct Sample {
    double measured_ratio;   // measured_ns / requested_ns (mean across blocks/iters)
    double error_ns;         // measured_ns - requested_ns
    int requested_ns;
    int num_blocks;
    int iters;
  };

  bool checkCorrectness(int requested_ns, int iters = 5,
                        double sane_max_factor = 1000.0) {
    auto deltas = collectDeltas(requested_ns, iters);
    if (deltas.empty()) {
      std::fprintf(stderr, "[SleepBench] no samples (timer non-monotonic?)\n");
      return false;
    }
    if (static_cast<int>(deltas.size()) <
        static_cast<int>(num_blocks_) * iters / 2) {
      std::fprintf(stderr,
                   "[SleepBench] too few monotonic samples: %zu / %d expected\n",
                   deltas.size(),
                   num_blocks_ * iters);
      return false;
    }
    uint64_t sum = 0;
    for (auto d : deltas) sum += d;
    double avg_ns = static_cast<double>(sum) / deltas.size();
    double max_allowed = static_cast<double>(requested_ns) * sane_max_factor +
                         100000.0;
    if (avg_ns > max_allowed) {
      std::fprintf(stderr,
                   "[SleepBench] avg measured %.0f ns >> requested %d ns "
                   "(>%.0fx, looks broken)\n",
                   avg_ns, requested_ns, sane_max_factor);
      return false;
    }
    return true;
  }

  Sample benchmark(int requested_ns, int iters = 20) {
    auto deltas = collectDeltas(requested_ns, iters);
    Sample s;
    s.requested_ns = requested_ns;
    s.num_blocks = num_blocks_;
    s.iters = iters;
    if (deltas.empty()) {
      s.measured_ratio = 0;
      s.error_ns = 0;
      return s;
    }
    uint64_t sum = 0;
    for (auto d : deltas) sum += d;
    double measured_ns = static_cast<double>(sum) / deltas.size();
    s.measured_ratio = measured_ns / static_cast<double>(requested_ns);
    s.error_ns = measured_ns - static_cast<double>(requested_ns);
    return s;
  }

 private:
  std::vector<uint64_t> collectDeltas(int requested_ns, int iters) {
    std::vector<uint64_t> deltas;
    deltas.reserve(static_cast<size_t>(iters) * num_blocks_);
    std::vector<uint64_t> host(2 * num_blocks_);

    for (int it = 0; it < iters; ++it) {
      CUDA_CHECK(cudaSetDevice(dev_));
      CUDA_CHECK(cudaMemsetAsync(d_ts_, 0, 2 * num_blocks_ * sizeof(uint64_t),
                                 stream_));
      nanosleep_probe_kernel<<<num_blocks_, 32, 0, stream_>>>(d_ts_,
                                                                requested_ns);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      CUDA_CHECK(cudaMemcpy(host.data(), d_ts_,
                            2 * num_blocks_ * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
      for (int b = 0; b < num_blocks_; ++b) {
        uint64_t s = host[b];
        uint64_t e = host[num_blocks_ + b];
        if (e > s) {
          deltas.push_back(e - s);
        }
      }
    }
    return deltas;
  }

  int dev_;
  int num_blocks_;
  uint64_t* d_ts_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

static void runTest() {
  int n_dev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_dev));
  if (n_dev < 1) {
    std::fprintf(stderr, "ERROR: no CUDA device found\n");
    std::printf("{\"Correctness\": \"FAIL\"}\n");
    std::exit(1);
  }

  const std::vector<int> requested_ns = {100, 1000, 10000, 100000, 1000000};
  const int num_blocks = 8;

  SleepBench bench(/*dev=*/0, num_blocks);

  bool ok = true;
  for (int ns : requested_ns) {
    if (!bench.checkCorrectness(ns)) {
      ok = false;
      break;
    }
  }

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", ok ? "PASS" : "FAIL");
  std::printf("  \"requested_ns_unit\": \"ns\",\n");
  std::printf("  \"measured_ratio_unit\": \"ratio\",\n");
  std::printf("  \"error_ns_unit\": \"ns\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < requested_ns.size(); ++i) {
    auto s = bench.benchmark(requested_ns[i]);
    std::printf(
        "    {\"requested_ns\": %d, \"measured_ratio\": %.4f, \"error_ns\": %.2f}%s\n",
        s.requested_ns, s.measured_ratio, s.error_ns,
        (i + 1 < requested_ns.size()) ? "," : "");
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
