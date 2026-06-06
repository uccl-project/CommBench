/*
 * Intra-node AllGather using the NCCL Device API (NCCL >= 2.29).
 *
 * A single fused device kernel uses LSA Multimem Copy building blocks to
 * broadcast each rank's send buffer into all ranks' receive buffers.
 *
 * Pre-requisite:
 *   export PATH=/home/uccl/miniconda3/envs/yihan/bin:/home/uccl/miniconda3/nvvm/bin:$PATH
 *   export LD_LIBRARY_PATH=/home/uccl/miniconda3/envs/yihan/lib:$LD_LIBRARY_PATH
 *
 * Compile:
 *   OMPI_CXX=g++-12 nvcc -O3 -std=c++17 -arch=sm_90 \
 *     --extended-lambda \
 *     -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE \
 *     -ccbin mpicxx \
 *     nccl_device_api_allgather_intra.cu -lnccl -o lsa_allgather_intra
 *
 * Run:
 *   mpirun --allow-run-as-root -np 8 ./lsa_allgather_intra 67108864 20
 * or mpirun --allow-run-as-root -np 8 ./lsa_allgather_intra
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CTA_COUNT 64
#define THREADS   256
#define UNROLL    16

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------
//
// Each rank provides `count` elements in its local `send` buffer.
// `recv` is a multimem pointer to a buffer of at least `count * nRanks`
// elements on every rank.
//
// Output layout on every rank:
//   recv[r * count : (r+1) * count] = rank r's send
//
// Each CTA should:
//   1. Compute its slice [start, start + n) from count / gridDim.x
//   2. Compute dst = r * count + start
//   3. Call ncclMultimemCopy to copy send[start..start+n) to recv[dst..dst+n)
//
// Hint: use ncclMultimemCopy<T, ncclCoopCta, size_t, UNROLL>(cta, src, dst, n)
template <typename T>
__global__ void allGatherImpl(
    T* send,
    T* recv,
    size_t count,
    ncclDevComm devComm) {
  // TODO
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------
//
// AllGather encapsulates:
//   - NCCL communicator init via MPI-broadcast of ncclUniqueId
//   - CUDA stream creation
//   - ncclDevComm creation with LSA requirements
//     (lsaBarrierCount=CTA_COUNT, lsaMultimem=true)
//   - ncclMemAlloc for send/recv buffers sized for maxCount
//   - ncclCommWindowRegister for recv buffer
//   - ncclGetLsaMultimemDevicePointer to get mcRecv_
//   - upload() / download() / run() / sync() / barrier() methods
//
class AllGather {
 public:
  AllGather(int rank, int nranks, size_t maxCount)
      : rank_(rank), nranks_(nranks), maxCount_(maxCount) {
    // TODO: Init NCCL comm via MPI-broadcast of ncclUniqueId
    // TODO: Create CUDA stream
    // TODO: Create ncclDevComm with lsaBarrierCount=CTA_COUNT, lsaMultimem=true
    // TODO: ncclMemAlloc sendPtr_ (maxCount_ floats)
    // TODO: ncclMemAlloc recvPtr_ (maxCount_ * nranks_ floats), zero it
    // TODO: ncclCommWindowRegister recvPtr_ -> recvWin_
    // TODO: ncclGetLsaMultimemDevicePointer recvWin_ -> mcRecv_
  }

  ~AllGather() {
    // TODO: Destroy all resources in reverse order
  }

  // Copy h into send buffer and zero the recv buffer for count elements.
  void upload(const std::vector<float>& h) {
    // TODO
  }

  // Copy count * nranks elements from recv buffer to host.
  std::vector<float> download(size_t count) {
    // TODO
  }

  void run(size_t count) {
    // TODO: Launch allGatherImpl<float> with CTA_COUNT blocks, THREADS threads
  }

  void sync()    { cudaStreamSynchronize(stream_); }
  void barrier() { MPI_Barrier(MPI_COMM_WORLD); }

  cudaStream_t stream() const { return stream_; }
  int rank()            const { return rank_; }
  int nranks()          const { return nranks_; }

 private:
  int          rank_, nranks_;
  size_t       maxCount_;
  ncclComm_t   comm_    = nullptr;
  ncclDevComm  devComm_{};
  cudaStream_t stream_  = nullptr;
  void*        sendPtr_ = nullptr;
  void*        recvPtr_ = nullptr;
  void*        mcRecv_  = nullptr;
  ncclWindow_t recvWin_{};
};

// ---------------------------------------------------------------------------
// Test + benchmark
// ---------------------------------------------------------------------------
struct Metric {
  size_t bytes;       // per-rank send bytes
  double latency_us;
  double algbw_GBs;
  double busbw_GBs;
};

static bool CheckCorrectness(AllGather& ag, const std::vector<float>& h) {
  const size_t count  = h.size();
  const int    nranks = ag.nranks();
  const int    rank   = ag.rank();

  ag.upload(h);
  cudaDeviceSynchronize();  // ensure memset is GPU-complete before barrier
  ag.barrier();
  ag.run(count);
  ag.sync();
  ag.barrier();
  cudaDeviceSynchronize();
  ag.barrier();

  bool ok = true;
  if (rank == 0) {
    auto out = ag.download(count);
    for (int src = 0; src < nranks; ++src) {
      float expected = (float)(src + 1);
      float got      = out[(size_t)src * count];
      if (got != expected) ok = false;
    }
  }
  return ok;
}

static Metric Benchmark(AllGather& ag, size_t count, int iters) {
  const int    nranks     = ag.nranks();
  const size_t send_bytes = count * sizeof(float);

  ag.barrier();
  for (int i = 0; i < 20; ++i) ag.run(count);
  ag.sync();
  ag.barrier();

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  cudaEventRecord(e0, ag.stream());
  for (int i = 0; i < iters; ++i) ag.run(count);
  cudaEventRecord(e1, ag.stream());
  cudaEventSynchronize(e1);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  ag.barrier();

  double us    = (ms * 1000.0) / (double)iters;
  double algbw = (double)send_bytes / (us * 1e3);
  double busbw = algbw * (double)(nranks - 1) / (double)nranks;
  return {send_bytes, us, algbw, busbw};
}

static void runTest(int rank, int nranks, size_t count, int iters) {
  std::vector<size_t> sizes_bytes = {
      (size_t)4   << 20,
      (size_t)16  << 20,
      (size_t)64  << 20,
      (size_t)128 << 20,
      (size_t)256 << 20,
  };
  if (count > 0) sizes_bytes = {count * sizeof(float)};

  size_t maxCount = sizes_bytes.back() / sizeof(float);
  AllGather ag(rank, nranks, maxCount);

  bool correctness = true;
  for (size_t b : sizes_bytes) {
    size_t cnt = b / sizeof(float);
    std::vector<float> h(cnt, (float)(rank + 1));
    correctness &= CheckCorrectness(ag, h);
  }

  std::vector<Metric> metrics;
  for (size_t b : sizes_bytes)
    metrics.push_back(Benchmark(ag, b / sizeof(float), iters));

  if (rank == 0) {
    printf("{\n");
    printf("  \"Correctness\": \"%s\",\n", correctness ? "PASS" : "FAIL");
    printf("  \"data_size_unit\": \"MB\",\n");
    printf("  \"algbw_unit\": \"GB/s\",\n");
    printf("  \"busbw_unit\": \"GB/s\",\n");
    printf("  \"latency_unit\": \"us\",\n");
    printf("  \"metrics\": [\n");
    for (size_t i = 0; i < metrics.size(); ++i) {
      printf("    {\"data_size\": %.3f, \"algbw\": %.3f, \"busbw\": %.3f, \"latency_avg\": %.3f}%s\n",
             (double)metrics[i].bytes / (1024.0 * 1024.0),
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

  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  int ngpus = 0;
  cudaGetDeviceCount(&ngpus);
  cudaSetDevice(rank % ngpus);

  size_t count = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 0;
  int    iters = (argc > 2) ? atoi(argv[2]) : 100;

  runTest(rank, nranks, count, iters);

  MPI_Finalize();
  return 0;
}
