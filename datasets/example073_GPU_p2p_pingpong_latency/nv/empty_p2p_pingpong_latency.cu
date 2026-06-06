// Two-GPU NVLink pingpong latency benchmark.
//
// AI completion task: implement the three system-scope memory primitives
// (`ld_acquire_sys`, `red_release_sys`, `wait_eq_sys`) and the
// `ping_pong_kernel` body marked with `// TODO`.  The driver, host
// orchestration (peer access setup, stream/event handling, timing,
// JSON output, sweep) is left intact.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__device__ __forceinline__ int ld_acquire_sys(int* ptr) {
  // TODO: emit `ld.global.acquire.sys.b32` and return the loaded int.
  return 0;
}

__device__ __forceinline__ void red_release_sys(int* ptr, int val) {
  // TODO: emit `fence.acq_rel.sys` followed by
  // `red.relaxed.sys.global.add.s32` to atomically add `val` to `*ptr`
  // with release semantics.
}

__device__ __forceinline__ void wait_eq_sys(int* flag, int expected) {
  // TODO: spin on `ld_acquire_sys(flag)` until it equals `expected`.
}

__global__ void ping_pong_kernel(int index, int num_flags, int* flags) {
  // TODO: implement the flag-flip pingpong:
  //   if index == 0:
  //     for i in [0, num_flags) step 2:
  //       red_release_sys(flags + i, 1);  (only thread 0)
  //       wait_eq_sys(flags + i + 1, 1);  (only thread 0)
  //   else:
  //     for i in [0, num_flags) step 2:
  //       wait_eq_sys(flags + i, 1);
  //       red_release_sys(flags + i + 1, 1);
  // Use __syncthreads() around the per-iteration memory ops if blockDim.x > 1.
}

class PingpongBench {
 public:
  PingpongBench(int dev_a, int dev_b, int flag_storage_dev, int max_flags)
      : dev_a_(dev_a), dev_b_(dev_b), flag_dev_(flag_storage_dev),
        max_flags_(max_flags) {
    enablePeer(dev_a_, dev_b_);
    enablePeer(dev_b_, dev_a_);

    CUDA_CHECK(cudaSetDevice(flag_dev_));
    CUDA_CHECK(cudaMalloc(&d_flags_, max_flags_ * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flags_, 0, max_flags_ * sizeof(int)));

    for (int d : {dev_a_, dev_b_}) {
      CUDA_CHECK(cudaSetDevice(d));
      CUDA_CHECK(cudaStreamCreate(&streams_[d == dev_a_ ? 0 : 1]));
      CUDA_CHECK(cudaEventCreate(&start_evt_[d == dev_a_ ? 0 : 1]));
      CUDA_CHECK(cudaEventCreate(&stop_evt_[d == dev_a_ ? 0 : 1]));
    }
  }

  ~PingpongBench() {
    if (d_flags_) {
      CUDA_CHECK(cudaSetDevice(flag_dev_));
      cudaFree(d_flags_);
    }
    for (int i = 0; i < 2; ++i) {
      if (streams_[i]) {
        cudaSetDevice(i == 0 ? dev_a_ : dev_b_);
        cudaStreamDestroy(streams_[i]);
        cudaEventDestroy(start_evt_[i]);
        cudaEventDestroy(stop_evt_[i]);
      }
    }
  }

  bool checkCorrectness(int num_flags, int num_threads) {
    resetFlags(num_flags);
    launchPair(num_flags, num_threads);
    syncBoth();

    std::vector<int> h(num_flags);
    CUDA_CHECK(cudaSetDevice(flag_dev_));
    CUDA_CHECK(cudaMemcpy(h.data(), d_flags_, num_flags * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_flags; ++i) {
      if (h[i] != 1) {
        std::fprintf(stderr,
                     "[PingpongBench] flag[%d] = %d (want 1) after %d pingpongs\n",
                     i, h[i], num_flags / 2);
        return false;
      }
    }
    return true;
  }

  struct Sample {
    double latency_us_per_pingpong;
    double throughput_mops;
    int num_pingpongs;
    int num_threads;
  };

  Sample benchmark(int num_flags, int num_threads, int repeats = 5,
                   int warmups = 2) {
    for (int w = 0; w < warmups; ++w) {
      resetFlags(num_flags);
      launchPair(num_flags, num_threads);
      syncBoth();
    }

    Sample best;
    best.latency_us_per_pingpong = 1e30;
    best.throughput_mops = 0.0;
    best.num_pingpongs = num_flags / 2;
    best.num_threads = num_threads;

    for (int r = 0; r < repeats; ++r) {
      resetFlags(num_flags);

      for (int i = 0; i < 2; ++i) {
        int d = (i == 0) ? dev_a_ : dev_b_;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaEventRecord(start_evt_[i], streams_[i]));
        ping_pong_kernel<<<1, num_threads, 0, streams_[i]>>>(i, num_flags,
                                                              d_flags_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_evt_[i], streams_[i]));
      }
      syncBoth();

      float ms[2] = {0.0f, 0.0f};
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(i == 0 ? dev_a_ : dev_b_));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], start_evt_[i], stop_evt_[i]));
      }
      double avg_ms = (ms[0] + ms[1]) / 2.0;
      double per_pp_us = avg_ms * 1000.0 / static_cast<double>(num_flags / 2);
      double mops = static_cast<double>(num_flags / 2) /
                    (avg_ms / 1000.0) / 1e6;

      if (per_pp_us < best.latency_us_per_pingpong) {
        best.latency_us_per_pingpong = per_pp_us;
        best.throughput_mops = mops;
      }
    }
    return best;
  }

 private:
  static void enablePeer(int from, int to) {
    int can = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can, from, to));
    if (!can) {
      std::fprintf(stderr,
                   "ERROR: P2P not available between dev %d and dev %d\n",
                   from, to);
      std::printf("{\"Correctness\": \"FAIL\"}\n");
      std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(from));
    cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr, "cudaDeviceEnablePeerAccess(%d -> %d): %s\n", from,
                   to, cudaGetErrorString(e));
      std::exit(EXIT_FAILURE);
    }
  }

  void resetFlags(int num_flags) {
    CUDA_CHECK(cudaSetDevice(flag_dev_));
    CUDA_CHECK(cudaMemset(d_flags_, 0, num_flags * sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void launchPair(int num_flags, int num_threads) {
    for (int i = 0; i < 2; ++i) {
      int d = (i == 0) ? dev_a_ : dev_b_;
      CUDA_CHECK(cudaSetDevice(d));
      ping_pong_kernel<<<1, num_threads, 0, streams_[i]>>>(i, num_flags,
                                                            d_flags_);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  void syncBoth() {
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(i == 0 ? dev_a_ : dev_b_));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  int dev_a_;
  int dev_b_;
  int flag_dev_;
  int max_flags_;
  int* d_flags_ = nullptr;
  cudaStream_t streams_[2] = {nullptr, nullptr};
  cudaEvent_t start_evt_[2] = {nullptr, nullptr};
  cudaEvent_t stop_evt_[2] = {nullptr, nullptr};
};

static void runTest() {
  int n_dev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_dev));
  if (n_dev < 2) {
    std::fprintf(stderr,
                 "ERROR: need >= 2 CUDA devices for P2P pingpong; found %d\n",
                 n_dev);
    std::printf("{\"Correctness\": \"FAIL\"}\n");
    std::exit(1);
  }

  const int dev_a = 0;
  const int dev_b = 1;
  const int flag_dev = dev_b;

  const std::vector<int> pingpong_counts = {32, 128, 512, 2048};
  const int max_flags = 2 * pingpong_counts.back();
  const int num_threads = 32;

  PingpongBench bench(dev_a, dev_b, flag_dev, max_flags);

  bool ok = true;
  for (int n_pp : pingpong_counts) {
    if (!bench.checkCorrectness(/*num_flags=*/2 * n_pp, num_threads)) {
      ok = false;
      break;
    }
  }

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", ok ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"pingpongs\",\n");
  std::printf("  \"throughput_unit\": \"Mops\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < pingpong_counts.size(); ++i) {
    int n_pp = pingpong_counts[i];
    auto s = bench.benchmark(/*num_flags=*/2 * n_pp, num_threads);
    std::printf(
        "    {\"pingpongs_num\": %d, \"latency_avg\": %.4f, "
        "\"throughput_avg\": %.4f, \"num_threads\": %d}%s\n",
        s.num_pingpongs, s.latency_us_per_pingpong, s.throughput_mops,
        s.num_threads, (i + 1 < pingpong_counts.size()) ? "," : "");
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
