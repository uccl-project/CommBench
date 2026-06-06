// This file implements a CUDA-based GPU peer-to-peer (P2P) communication benchmark.
// It measures correctness and performance of device-to-device memory copies
// between two GPUs across multiple data sizes.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

class GpuP2PComm {
 public:
  GpuP2PComm(int src = 0, int dst = 1)
      : srcDevice_(src), dstDevice_(dst) {
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount_));
    if (deviceCount_ < 2) std::exit(EXIT_FAILURE);
    if (srcDevice_ < 0 || srcDevice_ >= deviceCount_) std::exit(EXIT_FAILURE);
    if (dstDevice_ < 0 || dstDevice_ >= deviceCount_) std::exit(EXIT_FAILURE);

    checkP2PAccess_();
    if (p2pEnabled_) enableP2PAccess_();
  }

  ~GpuP2PComm() {
    cleanup_();
    if (p2pEnabled_) disableP2PAccess_();
  }

  void allocate(size_t bytes) {
    if (d_src_ && d_dst_ && allocatedBytes_ == bytes) return;
    cleanup_();

    CUDA_CHECK(cudaSetDevice(srcDevice_));
    CUDA_CHECK(cudaMalloc(&d_src_, bytes));
    CUDA_CHECK(cudaSetDevice(dstDevice_));
    CUDA_CHECK(cudaMalloc(&d_dst_, bytes));

    allocatedBytes_ = bytes;
  }

  void initialize(size_t bytes) {
    //TODO
  }

  void copy(size_t bytes) {
    //TODO
  }

  bool verify(size_t bytes) {
    const size_t n = bytes / sizeof(float);
    std::vector<float> h_src(n), h_dst(n);

    CUDA_CHECK(cudaSetDevice(srcDevice_));
    CUDA_CHECK(cudaMemcpy(h_src.data(), d_src_, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaSetDevice(dstDevice_));
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst_, bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
      if (h_src[i] != h_dst[i]) return false;
    }
    return true;
  }

  void cleanup() { cleanup_(); }

 private:
  int deviceCount_ = 0;
  int srcDevice_ = 0;
  int dstDevice_ = 1;

  void* d_src_ = nullptr;
  void* d_dst_ = nullptr;
  size_t allocatedBytes_ = 0;

  bool p2pEnabled_ = false;

  void checkP2PAccess_() {
    int canAccess = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, srcDevice_, dstDevice_));
    p2pEnabled_ = (canAccess != 0);
  }

  void enableP2PAccess_() {
   //TODO
  }

  void disableP2PAccess_() {
    CUDA_CHECK(cudaSetDevice(srcDevice_));
    cudaError_t e1 = cudaDeviceDisablePeerAccess(dstDevice_);
    if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessNotEnabled) {
      CUDA_CHECK(e1);
    }
    CUDA_CHECK(cudaSetDevice(dstDevice_));
    cudaError_t e2 = cudaDeviceDisablePeerAccess(srcDevice_);
    if (e2 != cudaSuccess && e2 != cudaErrorPeerAccessNotEnabled) {
      CUDA_CHECK(e2);
    }
  }

  void cleanup_() {
    if (d_src_) {
      CUDA_CHECK(cudaSetDevice(srcDevice_));
      CUDA_CHECK(cudaFree(d_src_));
      d_src_ = nullptr;
    }
    if (d_dst_) {
      CUDA_CHECK(cudaSetDevice(dstDevice_));
      CUDA_CHECK(cudaFree(d_dst_));
      d_dst_ = nullptr;
    }
    allocatedBytes_ = 0;
  }
};

// ── Test harness (correctness + performance benchmark) ──────────────────────

struct MetricRow {
  int data_size_mb = 0;
  double throughput_avg_gbps = 0.0;
  double latency_avg_us = 0.0;
  bool pass = false;
};

static std::vector<int> default_sizes_mb() {
  return {1, 2, 4, 8, 16, 32, 64, 128, 256};
}

static std::pair<bool, std::vector<MetricRow>> runTest(
    GpuP2PComm& comm,
    const std::vector<int>& sizes_mb,
    int warmup_iters,
    int iters) {

  std::vector<MetricRow> rows;
  rows.reserve(sizes_mb.size());
  bool overall_pass = true;

  for (int mb : sizes_mb) {
    const size_t bytes = static_cast<size_t>(mb) * 1024ULL * 1024ULL;

    comm.allocate(bytes);
    comm.initialize(bytes);

    // warmup (untimed)
    for (int i = 0; i < warmup_iters; ++i) {
      comm.copy(bytes);
    }

    // measured iterations with timing
    std::vector<float> times_ms;
    times_ms.reserve(std::max(1, iters));
    for (int i = 0; i < std::max(1, iters); ++i) {
      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));

      CUDA_CHECK(cudaEventRecord(start, 0));
      comm.copy(bytes);
      CUDA_CHECK(cudaEventRecord(stop, 0));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      times_ms.push_back(ms);

      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
    }

    const double avg_ms =
        std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
        static_cast<double>(times_ms.size());

    const bool pass = comm.verify(bytes);
    overall_pass = overall_pass && pass;

    MetricRow row;
    row.data_size_mb = mb;
    row.latency_avg_us = avg_ms * 1000.0;  // ms -> us

    const double sec = avg_ms / 1000.0;
    row.throughput_avg_gbps = (sec > 0.0)
        ? (static_cast<double>(bytes) * 8.0 / sec / 1e9)
        : 0.0;

    row.pass = pass;
    rows.push_back(row);

    comm.cleanup();
  }

  return {overall_pass, rows};
}

static void printJsonResult(bool overall_pass,
                            const std::vector<MetricRow>& rows) {
  std::cout << "{\n";
  std::cout << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL")
            << "\",\n";
  std::cout << "  \"data_size_unit\": \"MB\",\n";
  std::cout << "  \"throughput_unit\": \"Gbps\",\n";
  std::cout << "  \"latency_unit\": \"us\",\n";
  std::cout << "  \"metrics\": [\n";

  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::cout << "    {\"data_size\": " << r.data_size_mb
              << ", \"throughput_avg\": " << r.throughput_avg_gbps
              << ", \"latency_avg\": " << r.latency_avg_us << "}";
    if (i + 1 != rows.size()) std::cout << ",";
    std::cout << "\n";
  }

  std::cout << "  ]\n";
  std::cout << "}\n";
}

int main(int argc, char** argv) {
  int srcGpu = 0;
  int dstGpu = 1;
  int warmup = 5;
  int iters = 20;

  if (argc > 1) srcGpu = std::atoi(argv[1]);
  if (argc > 2) dstGpu = std::atoi(argv[2]);
  if (argc > 3) warmup = std::max(0, std::atoi(argv[3]));
  if (argc > 4) iters = std::max(1, std::atoi(argv[4]));

  std::vector<int> sizes;
  if (argc > 5) {
    for (int i = 5; i < argc; ++i) {
      int mb = std::atoi(argv[i]);
      if (mb > 0) sizes.push_back(mb);
    }
    if (sizes.empty()) sizes = default_sizes_mb();
  } else {
    sizes = default_sizes_mb();
  }

  GpuP2PComm comm(srcGpu, dstGpu);
  auto [overall_pass, rows] = runTest(comm, sizes, warmup, iters);

  printJsonResult(overall_pass, rows);
  return overall_pass ? 0 : 1;
}
