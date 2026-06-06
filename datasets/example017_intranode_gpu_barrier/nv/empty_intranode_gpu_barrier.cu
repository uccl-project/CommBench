// Intra-node GPU barrier using a shared atomic counter matrix.
// Each GPU owns one row of an NxN matrix (N = number of GPUs). To synchronize,
// each GPU atomically increments entries in its own row and decrements the
// corresponding entries in other GPUs' rows, then spins until its row drains
// to zero. This provides a device-side, cross-GPU synchronization primitive
// that avoids host-device round-trips.
//
// The program measures barrier latency across different GPU counts (2..N)
// and verifies correctness by having each GPU write a unique value before the
// barrier and read all values after.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(_e));                                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ── Device helpers ──────────────────────────────────────────────────────────

__device__ __forceinline__ int load_volatile(const int* p) {
  // TODO: Return a volatile load of *p to prevent caching.
  return 0;
}

// Device-side barrier across `nranks` GPUs.
// Must be called from a single warp (blockDim >= nranks, all in warp 0).
// The barrier is self-resetting: after completion all matrix entries return to
// their initial values, so it can be invoked repeatedly without host resets.
//
// Hint: The algorithm has three phases:
//   1. Publish writes with __threadfence_system + __syncthreads.
//   2. Each of the first `nranks` threads atomically increments its own row
//      entry (rows[rank][tid] += 1) and decrements the corresponding entry in
//      the remote GPU's row (rows[tid][rank] -= 1). Use atomicAdd_system /
//      atomicSub_system for cross-GPU visibility.
//   3. Spin until all first `nranks` bits in __ballot_sync report that their
//      row entry (rows[rank][tid]) has drained to <= 0.
__device__ void barrier_block(int** rows, int rank, int nranks) {
  // TODO: Implement the three-phase cross-GPU barrier described above.
}

// ── IntranodeGpuBarrier class ───────────────────────────────────────────────

class IntranodeGpuBarrier {
 public:
  IntranodeGpuBarrier() = default;
  ~IntranodeGpuBarrier() { teardown(); }

  // Allocate the NxN counter matrix and enable peer access among nranks GPUs.
  //
  // Hint:
  //   1. Enable peer access between every pair of GPUs (cudaDeviceEnablePeerAccess).
  //   2. Allocate one int[nranks] row per GPU using cudaMalloc, zero it.
  //   3. Copy the host row-pointer array to each GPU so every kernel has a
  //      device-accessible copy of the full table (d_tables_[rank]).
  void setup(int nranks) {
    // TODO: Implement barrier matrix allocation and peer access setup.
  }

  // Free all device memory and disable peer access.
  void teardown() {
    for (int i = 0; i < nranks_; i++) {
      cudaSetDevice(i);
      if (i < static_cast<int>(d_tables_.size()) && d_tables_[i])
        cudaFree(d_tables_[i]);
      if (i < static_cast<int>(h_rows_.size()) && h_rows_[i])
        cudaFree(h_rows_[i]);
    }
    d_tables_.clear();
    h_rows_.clear();

    for (int i = 0; i < nranks_; i++) {
      cudaSetDevice(i);
      for (int j = 0; j < nranks_; j++) {
        if (i == j) continue;
        cudaDeviceDisablePeerAccess(j);
        cudaGetLastError();
      }
    }
    nranks_ = 0;
  }

  // Zero all counter rows (needed between correctness and benchmark phases).
  void reset() {
    // TODO: cudaMemset each row to 0, then synchronize all GPUs.
  }

  int** getTable(int rank) const { return d_tables_[rank]; }
  int nranks() const { return nranks_; }

 private:
  int nranks_ = 0;
  std::vector<int*> h_rows_;
  std::vector<int**> d_tables_;
};

// ── Test kernels ────────────────────────────────────────────────────────────

// Each GPU writes (rank+1) to out[rank], barriers, then verifies the sum.
// On mismatch, atomicExch sets *pass to 0.
__global__ void correctness_kernel(int** rows, int rank, int nranks,
                                   int* out, int* pass) {
  // TODO: Thread 0 writes (rank + 1) into out[rank].
  // Then call barrier_block to synchronize all GPUs.
  // Thread 0 sums out[0..nranks-1] and compares to nranks*(nranks+1)/2.
  // On mismatch, atomicExch(pass, 0).
}

// Repeatedly invoke the barrier for timing purposes.
__global__ void benchmark_kernel(int** rows, int rank, int nranks, int niters) {
  // TODO: Call barrier_block in a loop `niters` times.
}

// ── Test harness ────────────────────────────────────────────────────────────

struct MetricRow {
  int nranks;
  double latency_avg_us;
  double throughput_avg;
  bool pass;
};

static void runTest(int max_gpus, std::vector<MetricRow>& results) {
  const int warmup_iters = 50;
  const int bench_iters = 200;

  IntranodeGpuBarrier barrier;

  for (int nranks = 2; nranks <= max_gpus; nranks++) {
    barrier.setup(nranks);

    // ── Correctness ──
    int* out = nullptr;
    int* pass_flag = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMallocManaged(&out, nranks * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&pass_flag, sizeof(int)));
    for (int i = 0; i < nranks; i++) out[i] = 0;
    *pass_flag = 1;

    for (int i = 0; i < nranks; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < nranks; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      correctness_kernel<<<1, 32>>>(barrier.getTable(i), i, nranks,
                                    out, pass_flag);
    }
    for (int i = 0; i < nranks; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    bool correct = (*pass_flag == 1);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(pass_flag));

    barrier.reset();

    // ── Warmup ──
    for (int i = 0; i < nranks; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      benchmark_kernel<<<1, 32>>>(barrier.getTable(i), i, nranks,
                                  warmup_iters);
    }
    for (int i = 0; i < nranks; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Timed run ──
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nranks; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      benchmark_kernel<<<1, 32>>>(barrier.getTable(i), i, nranks,
                                  bench_iters);
    }
    for (int i = 0; i < nranks; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_barrier_us = total_us / bench_iters;
    double kbarriers_per_sec =
        (per_barrier_us > 0) ? (1000.0 / per_barrier_us) : 0.0;

    MetricRow row;
    row.nranks = nranks;
    row.latency_avg_us = per_barrier_us;
    row.throughput_avg = kbarriers_per_sec;
    row.pass = correct;
    results.push_back(row);

    barrier.teardown();
  }
}

// ── main ────────────────────────────────────────────────────────────────────

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count < 2) {
    std::printf(
        "{\"Correctness\": \"SKIP\","
        " \"reason\": \"need at least 2 GPUs\"}\n");
    return 0;
  }

  int max_gpus = std::min(device_count, 8);

  std::vector<MetricRow> results;
  runTest(max_gpus, results);

  bool overall_pass = true;
  for (const auto& r : results) {
    if (!r.pass) overall_pass = false;
  }

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n",
              overall_pass ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"GPUs\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"throughput_unit\": \"Kbarriers/s\",\n");
  std::printf("  \"metrics\": [\n");

  for (size_t i = 0; i < results.size(); i++) {
    const auto& r = results[i];
    std::printf("    {\"data_size\": %d, \"latency_avg\": %.3f,"
                " \"throughput_avg\": %.3f}%s\n",
                r.nranks, r.latency_avg_us, r.throughput_avg,
                (i + 1 < results.size()) ? "," : "");
  }

  std::printf("  ]\n");
  std::printf("}\n");

  return overall_pass ? 0 : 1;
}
