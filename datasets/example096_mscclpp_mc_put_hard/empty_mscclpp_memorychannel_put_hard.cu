/*
Point-to-point MemoryChannel put() + signal() + wait() example.

High-level intent:
  Demonstrate the canonical producer-consumer synchronization pattern for
  mscclpp MemoryChannel. A bare put() pushes bytes into peer GPU memory, but
  the receiver has no ordering/completion signal unless the producer explicitly
  notifies it. This example adds signal() and wait() around put() so the
  consumer knows exactly when the remote write is ready to read.

Canonical form:
    one CTA. It is small, readable, and perfectly shows:
    put -> CTA sync -> signal
    wait -> CTA sync -> consume
    Intended for “minimal MSCCL++ MemoryChannel primitive.”
  Run with --gpus 2. Rank 0 is the producer and rank 1 is the consumer.
  Rank 0 uses MemoryChannel::put() to copy its local buffer into rank 1's
  registered memory, synchronizes the producer CTA, and then thread 0 calls
  signal(). Rank 1 has thread 0 spin in wait(), synchronizes the consumer CTA
  after the signal arrives, and then all CTA threads read/validate the buffer.

Multi-GPU stress form:
  With more GPUs, adjacent pairs run the same independent handshake
  concurrently: 0 -> 1, 2 -> 3, 4 -> 5, 6 -> 7. This reports aggregate
  peer-to-peer put+signal+wait throughput across all active pairs.

The host-side setup follows the known-working allreduce.cu scaffold in this
directory: forked one-process-per-rank launch, TcpBootstrap, CudaIpc registered
memory, and MemoryDevice2DeviceSemaphore-backed MemoryChannels.

From-scratch requirement (hard variant):
  The reference at ref_mscclpp_memorychannel_put.cu uses the MSCCL++ library
  (TcpBootstrap, Communicator, RegisteredMemory,
  Connection, MemoryChannel, MemoryDevice2DeviceSemaphore,
  DeviceHandle, UniqueId, ...). This empty file MUST NOT
  include any <mscclpp/*.hpp> header. The placeholder stub types below exist
  only so the file parses; the AI is expected to implement equivalent
  bootstrap / registered-memory / device-channel semantics from scratch
  (e.g., using CUDA IPC handles, POSIX SHM, Unix-domain-socket SCM_RIGHTS
  FD passing, and inline-PTX device-side put/signal/wait primitives) while
  preserving the JSON output schema and benchmark behavior of the reference.
*/

#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// =====================================================================
//   MSCCL++-equivalent stub types
//
//   The reference relies on these types from <mscclpp/core.hpp>,
//   <mscclpp/memory_channel.hpp>, and <mscclpp/gpu_utils.hpp>. The hard
//   variant forbids those includes, so the empty file provides minimal
//   local placeholders so signatures still parse. The AI must implement
//   the real semantics from scratch (or redesign these types entirely)
//   while keeping the public method surface and JSON output identical
//   to the reference.
// =====================================================================
namespace {

constexpr size_t UniqueIdBytes = 128;
using UniqueId = std::array<uint8_t, UniqueIdBytes>;

template <class T>
struct DeviceHandle {};

struct MemoryChannel {};
struct RegisteredMemory {};
struct Connection {};
struct MemoryDevice2DeviceSemaphore {};

struct TcpBootstrap {
  TcpBootstrap(int /*rank*/, int /*num_gpus*/) {}
  static UniqueId createUniqueId() { return {}; }
};

struct Communicator {
  explicit Communicator(std::shared_ptr<TcpBootstrap> /*bootstrap*/) {}
};

}  // namespace

__constant__ DeviceHandle<MemoryChannel> constMemChans[1];

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err__));                                  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)
#endif

static constexpr int kMaxGpus = 8;
static constexpr int kWarmupIters = 5;
static constexpr int kBenchIters = 100;
static constexpr int kCudaGraphLaunches = 15;
static constexpr int kPutBlocks = 1;
static constexpr int kPutThreads = 1024;
static constexpr int kSizesMiB[] = {1, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536};
static constexpr int kNumSizes = sizeof(kSizesMiB) / sizeof(kSizesMiB[0]);

struct Buffers {
  int num_gpus = 0;
  size_t elems = 0;
  size_t total_bytes = 0;
  std::shared_ptr<int> owned_buffer;
  int* buffer = nullptr;
  int* kernel_errors = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  RegisteredMemory registered_buffer;
  std::vector<Connection> memory_connections;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> semaphores;
  std::vector<MemoryChannel> memory_channels;
};

struct MetricRow {
  int data_size_mib;
  size_t actual_bytes;
  double throughput_avg_gb_per_s;
  double latency_avg_us;
  bool pass;
};

namespace {

std::string encode_unique_id(const UniqueId& id) {
  // TODO
}

UniqueId decode_unique_id(const std::string& hex) {
  // TODO
}

}  // namespace

__global__ void fill_buffer_kernel(int* ptr, size_t elems, int value) {
// TODO
}

__global__ void verify_buffer_kernel(const int* ptr, size_t elems, int expected, int* error_count) {
// TODO
}

extern "C" __global__ void __launch_bounds__(kPutThreads, 1)
    memorychannel_put_signal_wait_bench(int* buffer, size_t bytes, int rank, int num_gpus) {
        // TODO
}

extern "C" __global__ void __launch_bounds__(kPutThreads, 1)
    memorychannel_put_signal_wait_check(int* buffer, size_t bytes, int rank, int num_gpus, int* error_count) {
// TODO
}

static int putexample_count_active_pairs(int num_gpus) { return num_gpus / 2; }

class PutExample {
 public:
  PutExample(int rank, int num_gpus, const UniqueId& unique_id) : rank_(rank), num_gpus_(num_gpus) {
// TODO
  }

  Buffers make_buffers(size_t requested_total_bytes) {
// TODO
  }

  void fill_inputs(Buffers& buf) const {
// TODO
  }

  void launch(Buffers& buf, bool validate, bool check_launch_error = true) const {
// TODO
  }

  void launch_rank(Buffers& buf, int rank, bool validate, bool check_launch_error = true) const {
// TODO
  }

  void capture_graph(Buffers& buf, int graph_iters) const {
// TODO
  }

  void launch_graph(Buffers& buf) const {
// TODO
  }

  bool verify(Buffers& buf) const {
// TODO
  }

  bool all_ranks_passed(bool local_pass) const {
// TODO
  }

  double gather_avg_latency(double local_avg_us) const {
// TODO
  }

  double gather_max_latency(double local_avg_us) const {
// TODO
  }

  void sync_streams(Buffers& buf) const {
// TODO
  }

  void free_buffers(Buffers& buf) const {
// TODO
  }

  void barrier() const { 
    // TODO 
}
  int rank() const { 
    // TODO 
}
  bool is_root() const { 
    // TODO
}

 private:
  int rank_;
  int num_gpus_;
  std::shared_ptr<TcpBootstrap> bootstrap_;
  std::shared_ptr<Communicator> communicator_;
};

static void benchmark(PutExample& putexample) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int size_mib = kSizesMiB[i];
    const size_t requested_total_bytes = static_cast<size_t>(size_mib) << 20;

    Buffers buf = putexample.make_buffers(requested_total_bytes);

    if (i == 0) {
      putexample.fill_inputs(buf);
      putexample.barrier();
      putexample.launch(buf, true, true);
      putexample.sync_streams(buf);
      putexample.barrier();
      const bool single_pass = putexample.verify(buf);
      const bool single_all_pass = putexample.all_ranks_passed(single_pass);
      if (putexample.is_root()) {
        std::fprintf(stderr, "single-launch put+signal+wait check: %s\n", single_all_pass ? "PASS" : "FAIL");
      }
    }

    putexample.fill_inputs(buf);
    putexample.barrier();
    putexample.capture_graph(buf, kBenchIters);
    putexample.barrier();

    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      putexample.launch_graph(buf);
      putexample.sync_streams(buf);
      putexample.barrier();
    }

    putexample.barrier();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, buf.stream));
    for (int launch = 0; launch < kCudaGraphLaunches; ++launch) {
      putexample.launch_graph(buf);
    }
    CUDA_CHECK(cudaEventRecord(stop, buf.stream));
    putexample.sync_streams(buf);
    putexample.barrier();

    float local_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&local_total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double local_total_us = static_cast<double>(local_total_ms) * 1000.0;
    const double local_avg_us =
        local_total_us / static_cast<double>(kBenchIters) / static_cast<double>(kCudaGraphLaunches);
    const double avg_us = putexample.gather_max_latency(local_avg_us);
    const double avg_sec = avg_us / 1.0e6;
    const int active_pairs = putexample.rank() == 0 ? putexample_count_active_pairs(buf.num_gpus) : 0;
    const size_t aggregate_bytes = static_cast<size_t>(active_pairs) * buf.total_bytes;
    const double gb_per_s = avg_sec > 0.0 ? (static_cast<double>(aggregate_bytes) / avg_sec / 1.0e9) : 0.0;
    const bool pass = putexample.verify(buf);
    const bool all_pass = putexample.all_ranks_passed(pass);
    overall_pass = overall_pass && all_pass;

    if (putexample.is_root()) {
      rows.push_back(MetricRow{size_mib, aggregate_bytes, gb_per_s, avg_us, all_pass});
    }
    putexample.free_buffers(buf);
  }

  if (!putexample.is_root()) return;

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", overall_pass ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"MiB\",\n");
  std::printf("  \"actual_bytes_unit\": \"B\",\n");
  std::printf("  \"throughput_unit\": \"GB/s\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < rows.size(); ++i) {
    const MetricRow& row = rows[i];
    std::printf("    {\"data_size\": %d, \"actual_bytes\": %zu, \"throughput_avg\": %.3f, "
                "\"latency_avg\": %.3f, \"pass\": %s}",
                row.data_size_mib, row.actual_bytes, row.throughput_avg_gb_per_s, row.latency_avg_us,
                row.pass ? "true" : "false");
    if (i + 1 != rows.size()) std::printf(",");
    std::printf("\n");
  }
  std::printf("  ]\n");
  std::printf("}\n");
}

struct WorkerConfig {
  int rank = -1;
  int num_gpus = 0;
  UniqueId unique_id{};
};

static int run_worker(const WorkerConfig& config) {
  PutExample putexample(config.rank, config.num_gpus, config.unique_id);
  benchmark(putexample);
  return EXIT_SUCCESS;
}

static int launch_workers(const char* program_path, int num_gpus) {
  const std::string unique_id_hex = encode_unique_id(TcpBootstrap::createUniqueId());
  std::vector<pid_t> children;
  children.reserve(num_gpus);

  for (int rank = 0; rank < num_gpus; ++rank) {
    pid_t pid = fork();
    if (pid < 0) {
      std::perror("fork");
      return EXIT_FAILURE;
    }
    if (pid == 0) {
      std::string rank_str = std::to_string(rank);
      std::string gpus_str = std::to_string(num_gpus);
      execl(program_path, program_path, "--gpus", gpus_str.c_str(), "--rank", rank_str.c_str(), "--bootstrap-id",
            unique_id_hex.c_str(), static_cast<char*>(nullptr));
      std::perror("execl");
      _exit(EXIT_FAILURE);
    }
    children.push_back(pid);
  }

  int exit_code = EXIT_SUCCESS;
  for (pid_t child : children) {
    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
      std::perror("waitpid");
      exit_code = EXIT_FAILURE;
      continue;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      exit_code = EXIT_FAILURE;
    }
  }
  return exit_code;
}

int main(int argc, char** argv) {
  int requested_gpus = kMaxGpus;
  int rank = -1;
  std::string bootstrap_id_hex;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      std::string value = argv[++i];
      if (value != "kernel3") {
        std::fprintf(stderr, "Unknown --mode '%s'; this example only supports kernel3\n", value.c_str());
        return EXIT_FAILURE;
      }
    } else if (arg == "--gpus" && i + 1 < argc) {
      requested_gpus = std::atoi(argv[++i]);
    } else if (arg == "--rank" && i + 1 < argc) {
      rank = std::atoi(argv[++i]);
    } else if (arg == "--bootstrap-id" && i + 1 < argc) {
      bootstrap_id_hex = argv[++i];
    } else {
      std::fprintf(stderr, "Usage: %s [--gpus N]\n", argv[0]);
      return EXIT_FAILURE;
    }
  }

  int available_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
  const int num_gpus = std::min({requested_gpus, available_gpus, kMaxGpus});
  if (num_gpus <= 0) {
    std::fprintf(stderr, "No CUDA GPUs available\n");
    return EXIT_FAILURE;
  }

  if (rank < 0) {
    return launch_workers(argv[0], num_gpus);
  }

  if (bootstrap_id_hex.empty()) {
    std::fprintf(stderr, "Missing --bootstrap-id in worker mode\n");
    return EXIT_FAILURE;
  }

  WorkerConfig config;
  config.rank = rank;
  config.num_gpus = num_gpus;
  config.unique_id = decode_unique_id(bootstrap_id_hex);
  return run_worker(config);
}
