/*
 * Intra-node AllToAll using the NCCL Device API (NCCL >= 2.29).
 *
 * A single fused device kernel uses LSA Copy building blocks to send each
 * rank's per-destination chunk to the corresponding peer's receive buffer.
 *
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CTA_COUNT  64
#define THREADS    256
#define UNROLL     16

// ---------------------------------------------------------------------------
// Device kernel — one LSA Copy per CTA per destination.
// ---------------------------------------------------------------------------
//
// Each rank holds a send buffer laid out as [dst][chunk]. The receive buffer
// is laid out as [src][chunk]. Each CTA copies a slice of every destination
// chunk into the corresponding peer's recv buffer at offset [rank][slice].
//
// Params:
//   sendLocal — local send buffer (symmetric allocation)
//   recvWin   — symmetric recv window for LSA addressing
//   chunk     — elements per destination (total = chunk * nranks)
//   devComm   — NCCL device communicator handle
//
// Each CTA should:
//   1. Compute its slice [start, start + n) from chunk / gridDim.x
//   2. For each destination rank dst:
//      - srcPtr = sendLocal + dst * chunk + start
//      - dstPtr = ncclGetLsaPointer(recvWin, (rank * chunk + start) * sizeof(T), dst)
//      - call ncclLsaCopy with a dstLambda (nDst = 1)
//
// Hint: use the lambda overload of ncclLsaCopy
//   ncclLsaCopy<T, ncclCoopCta, decltype(dstLambda), size_t, UNROLL>(...)
template <typename T>
__global__ void customAllToAllKernel(
    T* sendLocal,
    ncclWindow_t recvWin,
    size_t chunk,
    ncclDevComm devComm) {
  // TODO
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------
//
// IntranodeAllToAll encapsulates:
//   - NCCL communicator init via MPI-broadcast of ncclUniqueId
//   - CUDA stream creation
//   - ncclDevComm creation with LSA requirements (lsaBarrierCount=CTA_COUNT)
//   - Run() method launching customAllToAllKernel with CTA_COUNT blocks
//     and THREADS threads
class IntranodeAllToAll {
 public:
  IntranodeAllToAll(int worldRank, int worldSize)
      : worldRank_(worldRank), worldSize_(worldSize) {
    // TODO: Init NCCL comm via MPI-broadcast of ncclUniqueId
    // TODO: Create CUDA stream
    // TODO: Create ncclDevComm with LSA requirements
  }

  ~IntranodeAllToAll() {
    // TODO: Destroy all resources created
  }

  void Run(float* sendLocal, ncclWindow_t recvWin, size_t chunk) {
    // TODO: Launch customAllToAllKernel
  }

  void Sync() { cudaStreamSynchronize(stream_); }

  ncclComm_t   comm()   const { return comm_; }
  cudaStream_t stream() const { return stream_; }
  int worldRank() const { return worldRank_; }
  int worldSize() const { return worldSize_; }

 private:
  int worldRank_, worldSize_;
  ncclComm_t   comm_   = nullptr;
  ncclDevComm  devComm_{};
  cudaStream_t stream_ = nullptr;
};

// ---------------------------------------------------------------------------
// Test + benchmark
// ---------------------------------------------------------------------------
struct Metric {
  size_t bytes;          // total bytes per rank (all destinations)
  double latency_us;
  double algbw_GBs;      // algorithm bandwidth per GPU (GB/s)
  double busbw_GBs;      // bus bandwidth per GPU (GB/s)
};

// CheckCorrectness should prepare test buffers, run the collective once,
// validate the received data, and aggregate the result across ranks.
static bool CheckCorrectness(IntranodeAllToAll& a2a, size_t totalCount) {
  // TODO
}

// Benchmark should allocate the necessary buffers, warm up the kernel,
// measure repeated launches, and return the timing/bandwidth metrics.
static Metric Benchmark(IntranodeAllToAll& a2a, size_t totalCount, int warmup,
                        int iters) {
  // TODO
}

static void runTest(int worldRank, int worldSize) {
  IntranodeAllToAll a2a(worldRank, worldSize);

  std::vector<size_t> sizes_bytes = {
      (size_t)4   << 20,
      (size_t)16  << 20,
      (size_t)64  << 20,
      (size_t)128 << 20,
      (size_t)256 << 20,
  };

  bool correctness = true;
  for (size_t b : sizes_bytes) {
    correctness &= CheckCorrectness(a2a, b / sizeof(float));
  }

  std::vector<Metric> metrics;
  for (size_t b : sizes_bytes) {
    metrics.push_back(Benchmark(a2a, b / sizeof(float), 20, 100));
  }

  if (worldRank == 0) {
    printf("{\n");
    printf("  \"Correctness\": \"%s\",\n", correctness ? "PASS" : "FAIL");
    printf("  \"data_size_unit\": \"MB\",\n");
    printf("  \"algbw_unit\": \"GB/s\",\n");
    printf("  \"busbw_unit\": \"GB/s\",\n");
    printf("  \"latency_unit\": \"us\",\n");
    printf("  \"metrics\": [\n");
    for (size_t i = 0; i < metrics.size(); ++i) {
      printf("    {\"data_size\": %zu, \"algbw\": %.3f, \"busbw\": %.3f, \"latency_avg\": %.3f}%s\n",
             metrics[i].bytes / (1024 * 1024),
             metrics[i].algbw_GBs,
             metrics[i].busbw_GBs,
             metrics[i].latency_us,
             (i + 1 == metrics.size()) ? "" : ",");
    }
    printf("  ]\n}\n");
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int worldRank = 0, worldSize = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int ngpus = 0;
  cudaGetDeviceCount(&ngpus);
  cudaSetDevice(worldRank % ngpus);

  runTest(worldRank, worldSize);

  MPI_Finalize();
  return 0;
}
