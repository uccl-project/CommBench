// Device-to-Host (D2H) copy bandwidth.
//
// AI completion task: implement the vectorised `copy_kernel` body and the
// `D2hBench::runOnce` mode dispatch marked `// TODO`.  Buffer setup,
// timing, correctness check, sweep, and JSON output are left intact.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

template <typename T = __half, typename VecType = uint4>
__global__ void copy_kernel(const T* __restrict__ src, T* __restrict__ dst,
                            size_t elems) {
  // TODO: stream `elems` of T from src to dst with VecType-wide
  // vectorised loads + stores.  Each thread strides by gridDim.x *
  // blockDim.x over the `elems / (sizeof(VecType)/sizeof(T))` vector
  // elements.  When `dst` is a UVA-mapped pinned host buffer the stores
  // traverse PCIe.
}

class D2hBench {
 public:
  using T = __half;

  D2hBench(int dev, size_t max_elems, int num_blocks = 16,
           int num_threads = 1024)
      : dev_(dev), max_elems_(max_elems), num_blocks_(num_blocks),
        num_threads_(num_threads) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaMalloc(&d_buf_, max_elems_ * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_, max_elems_ * sizeof(T)));
    h_pageable_ = static_cast<T*>(std::malloc(max_elems_ * sizeof(T)));
    if (!h_pageable_) {
      std::fprintf(stderr, "malloc pageable host buffer failed\n");
      std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));

    std::vector<T> tmp(max_elems_);
    for (size_t i = 0; i < max_elems_; ++i) {
      tmp[i] = static_cast<T>(static_cast<float>(i % 1024));
    }
    CUDA_CHECK(cudaMemcpy(d_buf_, tmp.data(), max_elems_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  ~D2hBench() {
    if (d_buf_) {
      cudaSetDevice(dev_);
      cudaFree(d_buf_);
    }
    if (h_pinned_) {
      cudaFreeHost(h_pinned_);
    }
    if (h_pageable_) {
      std::free(h_pageable_);
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
  }

  enum class Mode {
    MemcpyAsyncPageable,
    MemcpyAsyncPinned,
    KernelPinned,
  };

  bool checkCorrectness(size_t elems) {
    if (elems > max_elems_) {
      return false;
    }
    std::memset(h_pinned_, 0xFF, elems * sizeof(T));
    runOnce(elems, Mode::KernelPinned);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (size_t i = 0; i < elems; ++i) {
      float g = static_cast<float>(h_pinned_[i]);
      float w = static_cast<float>(i % 1024);
      if (g != w) {
        std::fprintf(stderr,
                     "[D2hBench] mismatch at %zu: got %f want %f (elems=%zu)\n",
                     i, g, w, elems);
        return false;
      }
    }
    return true;
  }

  struct Sample {
    double latency_us_per_iter;
    double throughput_gbps;
    size_t bytes;
    Mode mode;
  };

  Sample benchmark(size_t elems, Mode mode, int repeats = 5, int warmups = 2) {
    if (elems > max_elems_) {
      std::fprintf(stderr, "elems %zu > max_elems_ %zu\n", elems, max_elems_);
      std::exit(EXIT_FAILURE);
    }

    for (int w = 0; w < warmups; ++w) {
      runOnce(elems, mode);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    Sample best;
    best.latency_us_per_iter = 1e30;
    best.throughput_gbps = 0.0;
    best.bytes = elems * sizeof(T);
    best.mode = mode;

    for (int r = 0; r < repeats; ++r) {
      CUDA_CHECK(cudaSetDevice(dev_));
      CUDA_CHECK(cudaEventRecord(start_, stream_));
      runOnce(elems, mode);
      CUDA_CHECK(cudaEventRecord(stop_, stream_));
      CUDA_CHECK(cudaEventSynchronize(stop_));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));

      double per_iter_us = static_cast<double>(ms) * 1000.0;
      double gbps = static_cast<double>(best.bytes) /
                    (static_cast<double>(ms) / 1000.0) / 1e9;

      if (per_iter_us < best.latency_us_per_iter) {
        best.latency_us_per_iter = per_iter_us;
        best.throughput_gbps = gbps;
      }
    }
    return best;
  }

 private:
  void runOnce(size_t elems, Mode mode) {
    // TODO: dispatch one D2H copy on `stream_`:
    //   - Mode::MemcpyAsyncPageable → cudaMemcpyAsync(h_pageable_, d_buf_, ...)
    //   - Mode::MemcpyAsyncPinned   → cudaMemcpyAsync(h_pinned_,   d_buf_, ...)
    //   - Mode::KernelPinned        → launch copy_kernel<T,uint4> with
    //                                  num_blocks_ × num_threads_ threads
    //                                  writing into h_pinned_
    // Always cudaMemcpyDeviceToHost; remember to set the device and check
    // cudaGetLastError() after the kernel.
  }

  int dev_;
  size_t max_elems_;
  int num_blocks_;
  int num_threads_;
  T* d_buf_ = nullptr;
  T* h_pinned_ = nullptr;
  T* h_pageable_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

static const char* mode_str(D2hBench::Mode m) {
  switch (m) {
    case D2hBench::Mode::MemcpyAsyncPageable:
      return "cudaMemcpyAsync_pageable";
    case D2hBench::Mode::MemcpyAsyncPinned:
      return "cudaMemcpyAsync_pinned";
    case D2hBench::Mode::KernelPinned:
      return "kernel_pinned";
  }
  return "?";
}

static void runTest() {
  int n_dev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_dev));
  if (n_dev < 1) {
    std::fprintf(stderr, "ERROR: no CUDA device found\n");
    std::printf("{\"Correctness\": \"FAIL\"}\n");
    std::exit(1);
  }

  using T = __half;
  const std::vector<int> sizes_mb = {16, 64, 256, 1024};
  const int max_mb = sizes_mb.back();
  const size_t max_elems = static_cast<size_t>(max_mb) * 1024 * 1024 /
                           sizeof(T);

  D2hBench bench(/*dev=*/0, max_elems);

  bool ok = bench.checkCorrectness(/*elems=*/1u << 20);

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", ok ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"MB\",\n");
  std::printf("  \"throughput_unit\": \"GB/s\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < sizes_mb.size(); ++i) {
    int mb = sizes_mb[i];
    size_t elems = static_cast<size_t>(mb) * 1024 * 1024 / sizeof(T);
    auto pinned = bench.benchmark(elems, D2hBench::Mode::MemcpyAsyncPinned);
    auto page = bench.benchmark(elems, D2hBench::Mode::MemcpyAsyncPageable);
    auto kern = bench.benchmark(elems, D2hBench::Mode::KernelPinned);
    std::printf(
        "    {\"data_size\": %d, \"latency_avg\": %.4f, "
        "\"throughput_avg\": %.4f, \"throughput_pageable\": %.4f, "
        "\"throughput_kernel\": %.4f, \"latency_pageable\": %.4f, "
        "\"latency_kernel\": %.4f, \"mode\": \"%s\"}%s\n",
        mb, pinned.latency_us_per_iter, pinned.throughput_gbps,
        page.throughput_gbps, kern.throughput_gbps, page.latency_us_per_iter,
        kern.latency_us_per_iter, mode_str(D2hBench::Mode::MemcpyAsyncPinned),
        (i + 1 < sizes_mb.size()) ? "," : "");
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
