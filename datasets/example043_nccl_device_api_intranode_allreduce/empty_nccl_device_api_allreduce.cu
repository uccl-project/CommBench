/*
 * NCCL Device API intra-node AllReduce benchmark.
 *
 * Empty completion template.
 *
 * The MPI/NCCL setup, symmetric multicast windows, correctness checks,
 * benchmark, and JSON output are already wired up.  Complete only
 * NcclDeviceApiAllReduce::reduceShard(), keeping all class and function
 * signatures unchanged.
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(_e));                                  \
      MPI_Abort(MPI_COMM_WORLD, 1);                                           \
    }                                                                        \
  } while (0)

#define NCCL_CHECK(call)                                                     \
  do {                                                                       \
    ncclResult_t _r = (call);                                                 \
    if (_r != ncclSuccess) {                                                  \
      std::fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,     \
                   ncclGetErrorString(_r));                                  \
      MPI_Abort(MPI_COMM_WORLD, 1);                                           \
    }                                                                        \
  } while (0)

static constexpr int kCtaCount = 64;
static constexpr int kThreads = 256;
static constexpr int kUnroll = 16;

class NcclDeviceApiAllReduce {
 public:
  __device__ static void reduceShard(float* send, float* recv, size_t count,
                                     ncclDevComm dev_comm) {
    // TODO:
    // Use NCCL's device-side LSA team to split `count` elements across ranks.
    // Each rank should reduce its shard from the multicast send pointer and
    // copy the summed values to the matching offset in the multicast recv
    // pointer.  The reference solution uses ncclMultimemReduceSumCopy with
    // ncclCoopCta and kUnroll.
    (void)send;
    (void)recv;
    (void)count;
    (void)dev_comm;
  }
};

__global__ void allReduceKernel(float* send, float* recv, size_t count,
                                ncclDevComm dev_comm) {
  NcclDeviceApiAllReduce::reduceShard(send, recv, count, dev_comm);
}

struct MetricRow {
  double data_size_mib = 0.0;
  double latency_us = 0.0;
  double throughput_gbs = 0.0;
};

class NcclDeviceApiAllReduceRunner {
 public:
  NcclDeviceApiAllReduceRunner(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks_);

    CUDA_CHECK(cudaGetDeviceCount(&ngpus_));
    if (ngpus_ <= 0) {
      std::fprintf(stderr, "No CUDA devices found\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank_ % ngpus_));

    ncclUniqueId id;
    if (rank_ == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCL_CHECK(ncclCommInitRank(&comm_, nranks_, id, rank_));

    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kCtaCount;
    reqs.lsaMultimem = true;
    NCCL_CHECK(ncclDevCommCreate(comm_, &reqs, &dev_comm_));
  }

  NcclDeviceApiAllReduceRunner(const NcclDeviceApiAllReduceRunner&) = delete;
  NcclDeviceApiAllReduceRunner& operator=(
      const NcclDeviceApiAllReduceRunner&) = delete;

  ~NcclDeviceApiAllReduceRunner() {
    ncclDevCommDestroy(comm_, &dev_comm_);
    ncclCommDestroy(comm_);
    MPI_Finalize();
  }

  MetricRow runTest(size_t count, int iters, bool* correctness) const {
    int warmup = 20;
    size_t bytes = count * sizeof(float);
    void* sendbuf = nullptr;
    void* recvbuf = nullptr;
    void* mc_send = nullptr;
    void* mc_recv = nullptr;
    ncclWindow_t send_win;
    ncclWindow_t recv_win;

    NCCL_CHECK(ncclMemAlloc(&sendbuf, bytes));
    NCCL_CHECK(ncclMemAlloc(&recvbuf, bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm_, sendbuf, bytes, &send_win,
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm_, recvbuf, bytes, &recv_win,
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclGetLsaMultimemDevicePointer(send_win, 0, &mc_send));
    NCCL_CHECK(ncclGetLsaMultimemDevicePointer(recv_win, 0, &mc_recv));

    std::vector<float> host(count, static_cast<float>(rank_ + 1));
    CUDA_CHECK(cudaMemcpy(sendbuf, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recvbuf, 0, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup; ++i) {
      allReduceKernel<<<kCtaCount, kThreads, 0, stream>>>(
          static_cast<float*>(mc_send), static_cast<float*>(mc_recv), count,
          dev_comm_);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
      allReduceKernel<<<kCtaCount, kThreads, 0, stream>>>(
          static_cast<float*>(mc_send), static_cast<float*>(mc_recv), count,
          dev_comm_);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double latency_us = static_cast<double>(total_ms) * 1000.0 / iters;
    double throughput_gbs =
        (2.0 * static_cast<double>(bytes)) / (latency_us / 1.0e6) / 1.0e9;

    CUDA_CHECK(cudaMemcpy(host.data(), recvbuf, bytes, cudaMemcpyDeviceToHost));
    float expected = static_cast<float>(nranks_ * (nranks_ + 1) / 2);
    bool local_ok = true;
    size_t probes[] = {0, count / 2, count > 0 ? count - 1 : 0};
    for (size_t idx : probes) {
      if (count == 0) continue;
      if (std::fabs(host[idx] - expected) > 1.0e-3f) local_ok = false;
    }
    int ok_int = local_ok ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&ok_int, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    *correctness = all_ok == 1;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommWindowDeregister(comm_, send_win));
    NCCL_CHECK(ncclCommWindowDeregister(comm_, recv_win));
    NCCL_CHECK(ncclMemFree(sendbuf));
    NCCL_CHECK(ncclMemFree(recvbuf));

    MetricRow row;
    row.data_size_mib = static_cast<double>(bytes) / 1024.0 / 1024.0;
    row.latency_us = latency_us;
    row.throughput_gbs = throughput_gbs;
    return row;
  }

  bool isRoot() const { return rank_ == 0; }

 private:
  int rank_ = 0;
  int nranks_ = 1;
  int ngpus_ = 0;
  ncclComm_t comm_ = nullptr;
  ncclDevComm dev_comm_;
};

static void printJson(bool correctness, const std::vector<MetricRow>& metrics) {
  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", correctness ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"MiB\",\n");
  std::printf("  \"throughput_unit\": \"GB/s\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < metrics.size(); ++i) {
    const MetricRow& m = metrics[i];
    std::printf(
        "    {\"data_size\": %.2f, \"throughput_avg\": %.3f, "
        "\"latency_avg\": %.3f}%s\n",
        m.data_size_mib, m.throughput_gbs, m.latency_us,
        i + 1 == metrics.size() ? "" : ",");
  }
  std::printf("  ]\n");
  std::printf("}\n");
}

int main(int argc, char** argv) {
  int iters = (argc > 2) ? std::atoi(argv[2]) : 20;
  if (iters <= 0) iters = 1;

  // Sweep multiple element counts (default 1 MiB → 256 MiB) so metrics
  // carries more than one data_size entry. argv[1] pins to a single size.
  std::vector<size_t> counts;
  size_t single = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 0;
  if (single > 0) {
    counts.push_back(single);
  } else {
    counts = {
        262144ULL,    //   1 MiB
        1048576ULL,   //   4 MiB
        4194304ULL,   //  16 MiB
        16777216ULL,  //  64 MiB
        67108864ULL,  // 256 MiB
    };
  }

  NcclDeviceApiAllReduceRunner runner(argc, argv);
  bool correctness = true;
  std::vector<MetricRow> metrics;
  for (size_t count : counts) {
    bool case_ok = false;
    metrics.push_back(runner.runTest(count, iters, &case_ok));
    if (!case_ok) {
      correctness = false;
      break;
    }
  }

  if (runner.isRoot()) printJson(correctness, metrics);
  return correctness ? 0 : 1;
}
