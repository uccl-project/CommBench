/*
 * Intra-node ReduceScatter using the NCCL Device API (NCCL >= 2.29).
 *
 * A single fused device kernel calls the LSA Building Block
 * `ncclMultimemReduceSumCopy` once per CTA. Each rank owns one stripe of
 * the input ([r * chunk, (r + 1) * chunk)); the multimem source pointer
 * fans in the per-position sum across all LSA peers in hardware
 * (NVLink + NVSwitch multimem). The reduced stripe is written into the
 * rank's local recv buffer at offset 0, giving the ReduceScatter
 * semantics: each rank ends up with `chunk = count / lsaSize` reduced
 * elements at recv[0..chunk).
 *
 * Build (single host):
 *   OMPI_CXX=g++-12 nvcc -O3 -std=c++17 -arch=sm_90 \
 *       --extended-lambda \
 *       -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE \
 *       -ccbin mpicxx \
 *       empty_nccl_device_reducescatter.cu -lnccl -o lsa_reducescatter
 *
 * Run (single node, 8 GPUs):
 *   mpirun --allow-run-as-root -np 8 ./lsa_reducescatter
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CTA_COUNT  64
#define THREADS    256
#define UNROLL     16

// ---------------------------------------------------------------------------
// Device kernel — one LSA Building Block per CTA.
// ---------------------------------------------------------------------------
//
// Each rank reduces its own stripe and writes the reduced slice into the
// rank's local recv buffer (NOT the multimem recv pointer); that is what
// makes this a ReduceScatter instead of an AllReduce.
//
// Params:
//   mcSend    — multimem pointer to the symmetric send buffer
//   recvLocal — local (non-multimem) receive buffer
//   count     — total number of T elements across all ranks
//   devComm   — NCCL device communicator handle
//
// Each CTA should:
//   1. Obtain the LSA team via ncclTeamLsa(devComm)
//   2. Compute its own rank's stripe [begin, end) from count / nRanks
//   3. Divide the stripe across CTAs (gridDim.x)
//   4. Call ncclMultimemReduceSumCopy<T, ncclCoopCta, size_t, UNROLL>(...)
//      to reduce from mcSend into recvLocal
template <typename T>
__global__ void customReduceScatterKernel(
    T* mcSend,
    T* recvLocal,
    size_t count,
    ncclDevComm devComm) {
  // TODO
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------
//
// IntranodeReduceScatter encapsulates:
//   - NCCL communicator init via MPI-broadcast of ncclUniqueId
//   - CUDA stream creation
//   - ncclDevComm creation with LSA requirements (lsaBarrierCount=CTA_COUNT,
//     lsaMultimem=true)
//   - Run() method launching customReduceScatterKernel with CTA_COUNT blocks
//     and THREADS threads
class IntranodeReduceScatter {
 public:
  IntranodeReduceScatter(int worldRank, int worldSize)
      : worldRank_(worldRank), worldSize_(worldSize) {
    // TODO: Init NCCL comm via MPI-broadcast of ncclUniqueId
    // TODO: Create CUDA stream
    // TODO: Create ncclDevComm with LSA requirements
  }

  ~IntranodeReduceScatter() {
    // TODO: Destroy devComm, stream, comm
  }

  void Run(float* mcSend, float* recvLocal, size_t count) {
    // TODO: Launch customReduceScatterKernel
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
// Symmetric memory + window helpers
// ---------------------------------------------------------------------------
//
// SymBuf wraps a symmetric allocation (ncclMemAlloc) registered as a
// collective-symmetric window (NCCL_WIN_COLL_SYMMETRIC).
//
// SymAlloc: allocate `bytes` of symmetric memory and register a window.
// SymFree:  deregister window and free memory.
struct SymBuf {
  void*        ptr = nullptr;
  ncclWindow_t win{};
  size_t       bytes = 0;
};

static void SymAlloc(ncclComm_t comm, SymBuf& b, size_t bytes) {
  // TODO
}

static void SymFree(ncclComm_t comm, SymBuf& b) {
  // TODO
}

// ---------------------------------------------------------------------------
// Test + benchmark
// ---------------------------------------------------------------------------
struct Metric {
  size_t bytes;          // input bytes per rank
  double latency_us;
  double algbw_GBs;      // algorithm bandwidth per GPU (GB/s)
  double busbw_GBs;      // bus bandwidth per GPU (GB/s)
};

// CheckCorrectness:
//   1. SymAlloc send and recv buffers
//   2. Get multimem pointer via ncclGetLsaMultimemDevicePointer
//   3. Fill send buffer: rank r writes (r+1) + i*1e-6 at position i
//   4. Run ReduceScatter, verify each rank's output stripe matches
//      expected = sum_of_offsets + nranks * position * 1e-6
//   5. MPI_Allreduce to check all ranks passed
static bool CheckCorrectness(IntranodeReduceScatter& rs, size_t count) {
  // TODO
}

// Benchmark:
//   1. SymAlloc send and recv, get multimem pointer
//   2. Warmup iterations
//   3. Timed iterations using cudaEvent
//   4. Compute algbw = bytes / (us * 1e3) in GB/s
//   5. Compute busbw = bytes * (nranks-1) / nranks / (us * 1e3) in GB/s
static Metric Benchmark(IntranodeReduceScatter& rs, size_t count, int warmup,
                        int iters) {
  // TODO
}

static void runTest(int worldRank, int worldSize) {
  IntranodeReduceScatter rs(worldRank, worldSize);

  std::vector<size_t> sizes_bytes = {
      (size_t)4   << 20,
      (size_t)16  << 20,
      (size_t)64  << 20,
      (size_t)128 << 20,
      (size_t)256 << 20,
  };

  bool correctness = true;
  for (size_t b : sizes_bytes) {
    correctness &= CheckCorrectness(rs, b / sizeof(float));
  }

  std::vector<Metric> metrics;
  for (size_t b : sizes_bytes) {
    metrics.push_back(Benchmark(rs, b / sizeof(float), 20, 100));
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
