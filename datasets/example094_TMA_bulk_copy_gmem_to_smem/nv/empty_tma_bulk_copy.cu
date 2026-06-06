// Hopper/Blackwell TMA bulk-copy gmem→smem benchmark.
//
// AI completion task: implement the four mbarrier / TMA PTX wrappers
// (`mbar_init`, `mbar_expect_tx`, `mbar_wait_parity`, `cp_async_bulk_g2s`),
// the multi-CTA + multi-stage bench kernel body, and the one-tile
// correctness kernel body — all marked with `// TODO`.  Host-side buffer
// setup, smem opt-in, timing, sweep, and JSON output are left intact.

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

__device__ __forceinline__ void mbar_init(uint64_t* mbar, int count) {
  // TODO: emit `mbarrier.init.shared::cta.b64 [mbar], count`.
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar,
                                                uint32_t bytes) {
  // TODO: emit `mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], bytes`.
}

__device__ __forceinline__ void mbar_wait_parity(uint64_t* mbar,
                                                  uint32_t phase) {
  // TODO: spin on `mbarrier.try_wait.parity.shared::cta.b64 P, [mbar], phase`
  // until P is true.  See mirage's hopper/barrier.cuh wait() for the exact
  // PTX pattern.
}

// Fire one cp.async.bulk for `bytes` (multiple of 16) from `gmem_src` →
// `smem_dst`, crediting `mbar` via the mbarrier::complete_tx::bytes
// completion mechanism.  Caller is responsible for the prior expect_tx.
__device__ __forceinline__ void cp_async_bulk_g2s(void* smem_dst,
                                                   const void* gmem_src,
                                                   uint32_t bytes,
                                                   uint64_t* mbar) {
  // TODO: emit
  //   cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
  //     [smem_dst], [gmem_src], bytes, [mbar];
}

extern __shared__ uint8_t smem_pool[];

constexpr int kMaxStages = 8;

__global__ __launch_bounds__(128) void tma_bulk_load_bench_kernel(
    const uint8_t* __restrict__ gmem_src, uint32_t* __restrict__ sink,
    int tile_bytes, int iters_per_cta, int num_stages) {
  __shared__ alignas(8) uint64_t mbar[kMaxStages];

  // TODO: multi-CTA + multi-stage TMA-load pipeline:
  //   Per-CTA gmem slice base = blockIdx.x * iters_per_cta * tile_bytes.
  //   Smem layout: num_stages tile slots in a ring buffer; one mbarrier per
  //   stage (in `mbar[]`).
  //
  //   1. Thread 0 calls mbar_init(&mbar[s], 1) for s in [0, num_stages).
  //      __syncthreads().
  //   2. Thread 0 pre-issues the first min(num_stages, iters_per_cta)
  //      prefetches: for each stage s, mbar_expect_tx(&mbar[s], tile_bytes)
  //      and cp_async_bulk_g2s(smem_pool + s*tile_bytes,
  //                            cta_src + s*tile_bytes, tile_bytes, &mbar[s]).
  //   3. For it in [0, iters_per_cta):
  //        stage = it % num_stages
  //        phase = (it / num_stages) & 1
  //        all threads call mbar_wait_parity(&mbar[stage], phase)
  //        thread 0 reads smem_pool[stage*tile_bytes .. +4] as a uint32_t
  //        and adds it to a running accumulator
  //        if (it + num_stages < iters_per_cta), thread 0 issues the next
  //        prefetch for that stage (expect_tx + cp.async.bulk targeting the
  //        same smem slot, src = cta_src + (it+num_stages)*tile_bytes).
  //   4. Thread 0 writes the accumulator to sink[blockIdx.x].
}

__global__ __launch_bounds__(128) void tma_bulk_copy_one_tile_kernel(
    const uint8_t* __restrict__ gmem_src, uint8_t* __restrict__ gmem_dst,
    int tile_bytes) {
  __shared__ alignas(8) uint64_t mbar;

  // TODO: thread 0 inits mbar, posts expect_tx for tile_bytes, fires one
  // cp.async.bulk from gmem_src into smem_pool, all threads wait on the
  // barrier (phase 0), then the whole CTA copies smem_pool → gmem_dst with
  // uint4 vectorised stores.
}

class TmaBench {
 public:
  TmaBench(int dev, size_t total_bytes, int max_smem_bytes, int max_num_ctas)
      : dev_(dev), total_bytes_(total_bytes), max_num_ctas_(max_num_ctas) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaMalloc(&d_src_, total_bytes_));
    CUDA_CHECK(cudaMalloc(&d_dst_, total_bytes_));
    CUDA_CHECK(cudaMalloc(&d_sink_,
                          static_cast<size_t>(max_num_ctas_) * sizeof(uint32_t)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));

    CUDA_CHECK(cudaFuncSetAttribute(
        tma_bulk_load_bench_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        tma_bulk_copy_one_tile_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_bytes));

    std::vector<uint8_t> tmp(total_bytes_);
    for (size_t i = 0; i < total_bytes_; ++i) {
      tmp[i] = static_cast<uint8_t>(i * 37u + 11u);
    }
    CUDA_CHECK(cudaMemcpy(d_src_, tmp.data(), total_bytes_,
                          cudaMemcpyHostToDevice));
  }

  ~TmaBench() {
    if (d_src_) {
      cudaSetDevice(dev_);
      cudaFree(d_src_);
    }
    if (d_dst_) {
      cudaSetDevice(dev_);
      cudaFree(d_dst_);
    }
    if (d_sink_) {
      cudaSetDevice(dev_);
      cudaFree(d_sink_);
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
  }

  bool checkCorrectness(int tile_bytes) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaMemsetAsync(d_dst_, 0xFF, tile_bytes, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    tma_bulk_copy_one_tile_kernel<<<1, 128, tile_bytes, stream_>>>(
        d_src_, d_dst_, tile_bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<uint8_t> got(tile_bytes);
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst_, tile_bytes,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < tile_bytes; ++i) {
      uint8_t want = static_cast<uint8_t>(i * 37u + 11u);
      if (got[i] != want) {
        std::fprintf(stderr,
                     "[TmaBench] mismatch at byte %d: got %u want %u (tile=%d)\n",
                     i, got[i], want, tile_bytes);
        return false;
      }
    }
    return true;
  }

  struct Sample {
    double tile_latency_us;            // total_us / iters_per_cta
    double aggregate_throughput_gbps;  // total bytes / time, across all CTAs
    int tile_bytes;
    int num_stages;
    int num_ctas;
    int iters_per_cta;
    size_t total_bytes;
  };

  Sample benchmark(int num_ctas, int tile_bytes, int num_stages,
                   int iters_per_cta, int repeats = 5, int warmups = 2) {
    size_t needed = static_cast<size_t>(num_ctas) *
                    static_cast<size_t>(iters_per_cta) *
                    static_cast<size_t>(tile_bytes);
    if (needed > total_bytes_) {
      std::fprintf(stderr,
                   "[TmaBench] needed %zu B > allocated %zu B\n", needed,
                   total_bytes_);
      std::exit(EXIT_FAILURE);
    }
    if (num_ctas > max_num_ctas_) {
      std::fprintf(stderr, "[TmaBench] num_ctas %d > max %d\n", num_ctas,
                   max_num_ctas_);
      std::exit(EXIT_FAILURE);
    }
    if (num_stages > kMaxStages) {
      std::fprintf(stderr, "[TmaBench] num_stages %d > kMaxStages %d\n",
                   num_stages, kMaxStages);
      std::exit(EXIT_FAILURE);
    }

    for (int w = 0; w < warmups; ++w) {
      runOnce(num_ctas, tile_bytes, iters_per_cta, num_stages);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    Sample best;
    best.tile_latency_us = 1e30;
    best.aggregate_throughput_gbps = 0.0;
    best.tile_bytes = tile_bytes;
    best.num_stages = num_stages;
    best.num_ctas = num_ctas;
    best.iters_per_cta = iters_per_cta;
    best.total_bytes = needed;

    for (int r = 0; r < repeats; ++r) {
      CUDA_CHECK(cudaSetDevice(dev_));
      CUDA_CHECK(cudaEventRecord(start_, stream_));
      runOnce(num_ctas, tile_bytes, iters_per_cta, num_stages);
      CUDA_CHECK(cudaEventRecord(stop_, stream_));
      CUDA_CHECK(cudaEventSynchronize(stop_));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));

      double per_iter_us = static_cast<double>(ms) * 1000.0 /
                           static_cast<double>(iters_per_cta);
      double gbps = static_cast<double>(best.total_bytes) /
                    (static_cast<double>(ms) / 1000.0) / 1e9;

      if (gbps > best.aggregate_throughput_gbps) {
        best.tile_latency_us = per_iter_us;
        best.aggregate_throughput_gbps = gbps;
      }
    }
    return best;
  }

 private:
  void runOnce(int num_ctas, int tile_bytes, int iters_per_cta,
               int num_stages) {
    CUDA_CHECK(cudaSetDevice(dev_));
    int dyn_smem = num_stages * tile_bytes;
    tma_bulk_load_bench_kernel<<<num_ctas, 128, dyn_smem, stream_>>>(
        d_src_, d_sink_, tile_bytes, iters_per_cta, num_stages);
    CUDA_CHECK(cudaGetLastError());
  }

  int dev_;
  size_t total_bytes_;
  int max_num_ctas_;
  uint8_t* d_src_ = nullptr;
  uint8_t* d_dst_ = nullptr;
  uint32_t* d_sink_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

static void runTest() {
  int n_dev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_dev));
  if (n_dev < 1) {
    std::fprintf(stderr, "ERROR: no CUDA device found\n");
    std::printf("{\"Correctness\": \"FAIL\"}\n");
    std::exit(1);
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major < 9) {
    std::fprintf(stderr,
                 "ERROR: TMA bulk copy requires Hopper (sm_90+); detected sm_%d%d (%s)\n",
                 prop.major, prop.minor, prop.name);
    std::printf("{\"Correctness\": \"FAIL\"}\n");
    std::exit(1);
  }

  const int tile_bytes = 64 * 1024;
  const int num_stages = 3;
  const int iters_per_cta = 256;
  const int max_smem_bytes = num_stages * tile_bytes;

  std::vector<int> cta_sweep = {1, 4, 16, 64};
  cta_sweep.push_back(prop.multiProcessorCount);
  cta_sweep.push_back(prop.multiProcessorCount * 2);
  std::sort(cta_sweep.begin(), cta_sweep.end());
  cta_sweep.erase(std::unique(cta_sweep.begin(), cta_sweep.end()),
                  cta_sweep.end());

  const int max_num_ctas = cta_sweep.back();
  const size_t total_bytes = static_cast<size_t>(max_num_ctas) *
                             static_cast<size_t>(iters_per_cta) *
                             static_cast<size_t>(tile_bytes);

  TmaBench bench(/*dev=*/0, total_bytes, max_smem_bytes, max_num_ctas);

  bool ok = bench.checkCorrectness(tile_bytes);

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", ok ? "PASS" : "FAIL");
  std::printf("  \"tile_latency_unit\": \"us\",\n");
  std::printf("  \"aggregate_throughput_unit\": \"GB/s\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < cta_sweep.size(); ++i) {
    int n_ctas = cta_sweep[i];
    auto s = bench.benchmark(n_ctas, tile_bytes, num_stages, iters_per_cta);
    std::printf(
        "    {\"num_ctas\": %d, \"tile_latency_avg\": %.4f, "
        "\"aggregate_throughput\": %.4f, \"num_stages\": %d, "
        "\"tile_bytes\": %d, \"iters_per_cta\": %d, \"total_bytes\": %zu}%s\n",
        n_ctas, s.tile_latency_us, s.aggregate_throughput_gbps,
        s.num_stages, s.tile_bytes, s.iters_per_cta, s.total_bytes,
        (i + 1 < cta_sweep.size()) ? "," : "");
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
