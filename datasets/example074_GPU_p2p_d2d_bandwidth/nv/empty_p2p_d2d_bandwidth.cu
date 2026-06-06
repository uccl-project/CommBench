// NVLink GPU↔GPU D2D copy bandwidth.
//
// AI completion task: implement the vectorised `copy_kernel` body and the
// `CopyBench::runOnce` dispatch marked `// TODO`.  Peer-access setup,
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
  // elements.  `dst` is peer-mapped to a remote GPU; the stores will
  // traverse NVLink.
}

class CopyBench {
 public:
  using T = __half;

  CopyBench(int dev_src, int dev_dst, size_t max_elems, int num_blocks = 16,
            int num_threads = 1024)
      : dev_src_(dev_src), dev_dst_(dev_dst), max_elems_(max_elems),
        num_blocks_(num_blocks), num_threads_(num_threads) {
    enablePeer(dev_src_, dev_dst_);
    enablePeer(dev_dst_, dev_src_);

    CUDA_CHECK(cudaSetDevice(dev_src_));
    CUDA_CHECK(cudaMalloc(&d_src_, max_elems_ * sizeof(T)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));

    CUDA_CHECK(cudaSetDevice(dev_dst_));
    CUDA_CHECK(cudaMalloc(&d_dst_, max_elems_ * sizeof(T)));

    std::vector<T> host(max_elems_);
    for (size_t i = 0; i < max_elems_; ++i) {
      host[i] = static_cast<T>(static_cast<float>(i % 1024));
    }
    CUDA_CHECK(cudaSetDevice(dev_src_));
    CUDA_CHECK(cudaMemcpy(d_src_, host.data(), max_elems_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  ~CopyBench() {
    if (d_src_) {
      cudaSetDevice(dev_src_);
      cudaFree(d_src_);
    }
    if (d_dst_) {
      cudaSetDevice(dev_dst_);
      cudaFree(d_dst_);
    }
    if (stream_) {
      cudaSetDevice(dev_src_);
      cudaStreamDestroy(stream_);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
  }

  enum class Mode { KernelPush, MemcpyAsync };

  bool checkCorrectness(size_t elems) {
    if (elems > max_elems_) {
      return false;
    }
    CUDA_CHECK(cudaSetDevice(dev_dst_));
    CUDA_CHECK(cudaMemsetAsync(d_dst_, 0xFF, elems * sizeof(T), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    runOnce(elems, Mode::KernelPush);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<T> got(elems);
    CUDA_CHECK(cudaSetDevice(dev_dst_));
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst_, elems * sizeof(T),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
      float g = static_cast<float>(got[i]);
      float w = static_cast<float>(i % 1024);
      if (g != w) {
        std::fprintf(stderr,
                     "[CopyBench] mismatch at %zu: got %f want %f (elems=%zu)\n",
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
      CUDA_CHECK(cudaSetDevice(dev_src_));
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

  void runOnce(size_t elems, Mode mode) {
    // TODO: dispatch a single copy on `stream_`:
    //   - Mode::KernelPush     → launch copy_kernel<T,uint4> with
    //                            num_blocks_ blocks × num_threads_ threads
    //   - Mode::MemcpyAsync    → cudaMemcpyAsync(d_dst_, d_src_,
    //                            elems * sizeof(T), cudaMemcpyDeviceToDevice,
    //                            stream_);
    // Remember to set the source device and check cudaGetLastError().
  }

  int dev_src_;
  int dev_dst_;
  size_t max_elems_;
  int num_blocks_;
  int num_threads_;
  T* d_src_ = nullptr;
  T* d_dst_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

static const char* mode_str(CopyBench::Mode m) {
  return m == CopyBench::Mode::KernelPush ? "kernel_push" : "cudaMemcpyAsync";
}

static void runTest() {
  int n_dev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_dev));
  if (n_dev < 2) {
    std::fprintf(stderr,
                 "ERROR: need >= 2 CUDA devices for P2P D2D bandwidth; found %d\n",
                 n_dev);
    std::printf("{\"Correctness\": \"FAIL\"}\n");
    std::exit(1);
  }

  const int dev_src = 0;
  const int dev_dst = 1;
  using T = __half;

  const std::vector<int> sizes_mb = {16, 64, 256, 1024};
  const int max_mb = sizes_mb.back();
  const size_t max_elems = static_cast<size_t>(max_mb) * 1024 * 1024 /
                           sizeof(T);

  CopyBench bench(dev_src, dev_dst, max_elems);

  bool ok = bench.checkCorrectness(/*elems=*/1u << 20);

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", ok ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"MB\",\n");
  std::printf("  \"throughput_unit\": \"GB/s\",\n");
  std::printf("  \"throughput_cuda_memcpy_unit\": \"GB/s\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"latency_cuda_memcpy_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < sizes_mb.size(); ++i) {
    int mb = sizes_mb[i];
    size_t elems = static_cast<size_t>(mb) * 1024 * 1024 / sizeof(T);
    auto k = bench.benchmark(elems, CopyBench::Mode::KernelPush);
    auto m = bench.benchmark(elems, CopyBench::Mode::MemcpyAsync);
    std::printf(
        "    {\"data_size\": %d, \"latency_avg\": %.4f, "
        "\"throughput_avg\": %.4f, \"latency_cuda_memcpy\": %.4f, "
        "\"throughput_cuda_memcpy\": %.4f, \"mode\": \"%s\"}%s\n",
        mb, k.latency_us_per_iter, k.throughput_gbps, m.latency_us_per_iter,
        m.throughput_gbps, mode_str(CopyBench::Mode::KernelPush),
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
