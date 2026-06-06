/*
All-pairs MemoryChannel relaxedSignal() + wait() + warp-striped put() benchmark.

High-level intent:
  Demonstrate the barrier pattern used by allgather-style collectives:
  every rank tells every peer it is ready with relaxedSignal(), waits for
  every peer, then pushes its local slice into every peer's output buffer.

Canonical form:
  One CTA per rank. Warps first distribute the all-pairs relaxedSignal/wait
  barrier. After a CTA sync, the same warps stripe put() work across peers so
  all peer links can make progress together.

Why relaxedSignal():
  The input slice is filled before the benchmark kernel is launched. Inside
  the kernel there are no producer writes that need a release fence before the
  signal, so relaxedSignal() is enough for the readiness barrier.

Example build/run from this directory:
    python build_and_run.py --source ref_mscclpp_relaxedsingal.cu --gpus 8
*/

#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;

static constexpr int kMaxGpus = 8;
static constexpr int kMaxPeers = kMaxGpus - 1;
static constexpr int kWarpSize = 32;
static constexpr int kWarmupIters = 5;
static constexpr int kBenchIters = 100;
static constexpr int kCudaGraphLaunches = 15;
static constexpr int kBlocks = 1;
static constexpr int kThreads = 512;
static constexpr int kSizesMiB[] = {1, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536};
static constexpr int kNumSizes = sizeof(kSizesMiB) / sizeof(kSizesMiB[0]);

__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemChans[kMaxPeers];

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

struct Buffers {
  int num_gpus = 0;
  int num_peers = 0;
  size_t elems_per_rank = 0;
  size_t bytes_per_rank = 0;
  size_t total_bytes = 0;
  std::shared_ptr<int> owned_buffer;
  int* buffer = nullptr;
  int* kernel_errors = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  mscclpp::RegisteredMemory registered_buffer;
  std::vector<int> peer_ranks;
  std::vector<mscclpp::Connection> memory_connections;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> semaphores;
  std::vector<mscclpp::MemoryChannel> memory_channels;
};

struct MetricRow {
  int data_size_mib;
  size_t actual_bytes;
  double throughput_avg_gb_per_s;
  double latency_avg_us;
  bool pass;
};

namespace {

std::string encode_unique_id(const mscclpp::UniqueId& id) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(id.size() * 2);
  for (uint8_t byte : id) {
    out.push_back(kHex[(byte >> 4) & 0xF]);
    out.push_back(kHex[byte & 0xF]);
  }
  return out;
}

mscclpp::UniqueId decode_unique_id(const std::string& hex) {
  if (hex.size() != mscclpp::UniqueIdBytes * 2) {
    throw std::runtime_error("Invalid bootstrap unique id length");
  }
  auto hex_value = [](char c) -> uint8_t {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + c - 'a');
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + c - 'A');
    throw std::runtime_error("Invalid bootstrap unique id hex digit");
  };

  mscclpp::UniqueId id{};
  for (size_t i = 0; i < id.size(); ++i) {
    id[i] = static_cast<uint8_t>((hex_value(hex[2 * i]) << 4) | hex_value(hex[2 * i + 1]));
  }
  return id;
}

}  // namespace

__global__ void fill_allgather_buffer_kernel(int* ptr, size_t elems_per_rank, int rank, int num_gpus) {
      // TODO
}

__global__ void verify_allgather_buffer_kernel(const int* ptr, size_t elems_per_rank, int num_gpus, int* error_count) {
      // TODO
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
    relaxed_signal_allgather_bench(size_t bytes_per_rank, int rank, int num_gpus) {
      // TODO
}

class mscclpp_relaxedsignal {
 public:
  mscclpp_relaxedsignal(int rank, int num_gpus, const mscclpp::UniqueId& unique_id)
      : rank_(rank), num_gpus_(num_gpus) {
      // TODO
  }

  Buffers make_buffers(size_t requested_bytes_per_rank) {
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
  int num_gpus() const { 
          // TODO
  }
  bool is_root() const { 
          // TODO 
  }

 private:
  int rank_;
  int num_gpus_;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap_;
  std::shared_ptr<mscclpp::Communicator> communicator_;
};

static void benchmark(mscclpp_relaxedsignal& relaxedsignal) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int size_mib = kSizesMiB[i];
    const size_t requested_bytes_per_rank = static_cast<size_t>(size_mib) << 20;

    Buffers buf = relaxedsignal.make_buffers(requested_bytes_per_rank);

    if (i == 0) {
      relaxedsignal.fill_inputs(buf);
      relaxedsignal.barrier();
      relaxedsignal.launch(buf, true, true);
      relaxedsignal.sync_streams(buf);
      relaxedsignal.barrier();
      const bool single_pass = relaxedsignal.verify(buf);
      const bool single_all_pass = relaxedsignal.all_ranks_passed(single_pass);
      if (relaxedsignal.is_root()) {
        std::fprintf(stderr, "single-launch relaxedSignal+wait+put check: %s\n",
                     single_all_pass ? "PASS" : "FAIL");
      }
    }

    relaxedsignal.fill_inputs(buf);
    relaxedsignal.barrier();
    relaxedsignal.capture_graph(buf, kBenchIters);
    relaxedsignal.barrier();

    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      relaxedsignal.launch_graph(buf);
      relaxedsignal.sync_streams(buf);
      relaxedsignal.barrier();
    }

    relaxedsignal.barrier();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, buf.stream));
    for (int launch = 0; launch < kCudaGraphLaunches; ++launch) {
      relaxedsignal.launch_graph(buf);
    }
    CUDA_CHECK(cudaEventRecord(stop, buf.stream));
    relaxedsignal.sync_streams(buf);
    relaxedsignal.barrier();

    float local_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&local_total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double local_total_us = static_cast<double>(local_total_ms) * 1000.0;
    const double local_avg_us =
        local_total_us / static_cast<double>(kBenchIters) / static_cast<double>(kCudaGraphLaunches);
    const double avg_us = relaxedsignal.gather_max_latency(local_avg_us);
    const double avg_sec = avg_us / 1.0e6;
    const size_t aggregate_bytes =
        static_cast<size_t>(relaxedsignal.num_gpus()) * static_cast<size_t>(relaxedsignal.num_gpus() - 1) *
        buf.bytes_per_rank;
    const double gb_per_s = avg_sec > 0.0 ? (static_cast<double>(aggregate_bytes) / avg_sec / 1.0e9) : 0.0;
    const bool pass = relaxedsignal.verify(buf);
    const bool all_pass = relaxedsignal.all_ranks_passed(pass);
    overall_pass = overall_pass && all_pass;

    if (relaxedsignal.is_root()) {
      rows.push_back(MetricRow{size_mib, aggregate_bytes, gb_per_s, avg_us, all_pass});
    }
    relaxedsignal.free_buffers(buf);
  }

  if (!relaxedsignal.is_root()) return;

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", overall_pass ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"MiB/rank\",\n");
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
  mscclpp::UniqueId unique_id{};
};

static int run_worker(const WorkerConfig& config) {
  mscclpp_relaxedsignal relaxedsignal(config.rank, config.num_gpus, config.unique_id);
  benchmark(relaxedsignal);
  return EXIT_SUCCESS;
}

static int launch_workers(const char* program_path, int num_gpus) {
  const std::string unique_id_hex = encode_unique_id(mscclpp::TcpBootstrap::createUniqueId());
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
  if (num_gpus < 2) {
    std::fprintf(stderr, "Need at least 2 CUDA GPUs\n");
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
