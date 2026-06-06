// Single-GPU HBM copy bandwidth at 4B / 8B / 16B vectorisation widths.
//
// AI completion task: implement the vectorised `copy_kernel` body and the
// `VecCopyBench::runOnce` width dispatch marked `// TODO`.  Buffer setup,
// timing, correctness, sweep, and JSON output are left intact.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

template <typename T, typename VecType>
__global__ void copy_kernel(const T* __restrict__ src, T* __restrict__ dst,
                            size_t elems) {
  // TODO: stream `elems` of T from src to dst with VecType-wide vectorised
  // loads + stores.  vec_elems = elems / (sizeof(VecType)/sizeof(T)); each
  // thread strides by gridDim.x * blockDim.x.
}

class VecCopyBench {
 public:
  using T = __half;
  enum class Width { B4, B8, B16 };

  VecCopyBench(int dev, size_t max_elems, int num_blocks = 32,
               int num_threads = 1024)
      : dev_(dev), max_elems_(max_elems), num_blocks_(num_blocks),
        num_threads_(num_threads) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaMalloc(&d_src_, max_elems_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dst_, max_elems_ * sizeof(T)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));

    std::vector<T> tmp(max_elems_);
    for (size_t i = 0; i < max_elems_; ++i) {
      tmp[i] = static_cast<T>(static_cast<float>(i % 1024));
    }
    CUDA_CHECK(cudaMemcpy(d_src_, tmp.data(), max_elems_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  ~VecCopyBench() {
    if (d_src_) {
      cudaSetDevice(dev_);
      cudaFree(d_src_);
    }
    if (d_dst_) {
      cudaSetDevice(dev_);
      cudaFree(d_dst_);
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
  }

  bool checkCorrectness(size_t elems, Width width) {
    if (elems > max_elems_) {
      return false;
    }
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaMemsetAsync(d_dst_, 0xFF, elems * sizeof(T), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    runOnce(elems, width);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<T> got(elems);
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst_, elems * sizeof(T),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
      float g = static_cast<float>(got[i]);
      float w = static_cast<float>(i % 1024);
      if (g != w) {
        std::fprintf(stderr,
                     "[VecCopyBench] mismatch at %zu (width=%d): got %f want %f\n",
                     i, static_cast<int>(width), g, w);
        return false;
      }
    }
    return true;
  }

  struct Sample {
    double latency_us_per_iter;
    double throughput_gbps;
    size_t bytes;
    Width width;
  };

  Sample benchmark(size_t elems, Width width, int repeats = 5,
                   int warmups = 2) {
    if (elems > max_elems_) {
      std::fprintf(stderr, "elems %zu > max_elems_ %zu\n", elems, max_elems_);
      std::exit(EXIT_FAILURE);
    }

    for (int w = 0; w < warmups; ++w) {
      runOnce(elems, width);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    Sample best;
    best.latency_us_per_iter = 1e30;
    best.throughput_gbps = 0.0;
    best.bytes = elems * sizeof(T);
    best.width = width;

    for (int r = 0; r < repeats; ++r) {
      CUDA_CHECK(cudaSetDevice(dev_));
      CUDA_CHECK(cudaEventRecord(start_, stream_));
      runOnce(elems, width);
      CUDA_CHECK(cudaEventRecord(stop_, stream_));
      CUDA_CHECK(cudaEventSynchronize(stop_));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));

      double per_iter_us = static_cast<double>(ms) * 1000.0;
      double bw_gbps = 2.0 * static_cast<double>(best.bytes) /
                       (static_cast<double>(ms) / 1000.0) / 1e9;

      if (per_iter_us < best.latency_us_per_iter) {
        best.latency_us_per_iter = per_iter_us;
        best.throughput_gbps = bw_gbps;
      }
    }
    return best;
  }

 private:
  void runOnce(size_t elems, Width width) {
    // TODO: dispatch copy_kernel<T, VecType> on `stream_` with
    // num_blocks_ × num_threads_ threads, picking VecType based on `width`:
    //   Width::B4  -> uint    (4 bytes)
    //   Width::B8  -> uint2   (8 bytes)
    //   Width::B16 -> uint4   (16 bytes)
    // Remember to set the device and check cudaGetLastError().
  }

  int dev_;
  size_t max_elems_;
  int num_blocks_;
  int num_threads_;
  T* d_src_ = nullptr;
  T* d_dst_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

static int width_bytes(VecCopyBench::Width w) {
  switch (w) {
    case VecCopyBench::Width::B4:  return 4;
    case VecCopyBench::Width::B8:  return 8;
    case VecCopyBench::Width::B16: return 16;
  }
  return 0;
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

  VecCopyBench bench(/*dev=*/0, max_elems);

  bool ok = true;
  for (auto w : {VecCopyBench::Width::B4, VecCopyBench::Width::B8,
                 VecCopyBench::Width::B16}) {
    if (!bench.checkCorrectness(/*elems=*/1u << 20, w)) {
      ok = false;
      break;
    }
  }

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", ok ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"MB\",\n");
  std::printf("  \"throughput_unit\": \"GB/s\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < sizes_mb.size(); ++i) {
    int mb = sizes_mb[i];
    size_t elems = static_cast<size_t>(mb) * 1024 * 1024 / sizeof(T);
    auto b4  = bench.benchmark(elems, VecCopyBench::Width::B4);
    auto b8  = bench.benchmark(elems, VecCopyBench::Width::B8);
    auto b16 = bench.benchmark(elems, VecCopyBench::Width::B16);
    std::printf(
        "    {\"data_size\": %d, \"latency_avg\": %.4f, "
        "\"throughput_avg\": %.4f, "
        "\"throughput_4B\": %.4f, \"throughput_8B\": %.4f, "
        "\"latency_4B\": %.4f, \"latency_8B\": %.4f, "
        "\"width_bytes\": %d}%s\n",
        mb, b16.latency_us_per_iter, b16.throughput_gbps,
        b4.throughput_gbps, b8.throughput_gbps,
        b4.latency_us_per_iter, b8.latency_us_per_iter,
        width_bytes(VecCopyBench::Width::B16),
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
