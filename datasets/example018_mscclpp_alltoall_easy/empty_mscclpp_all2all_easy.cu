// implement the fastest All to All CUDA kernel
// intra node of 8 B300 GPUs
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <mscclpp/core.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/semaphore.hpp>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemChans[512];

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
static constexpr int kAllToAllBlocks = 64;
static constexpr int kAllToAllThreads = 1024;
static constexpr int kSizesMiB[] = {1, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536};
static constexpr int kNumSizes = sizeof(kSizesMiB) / sizeof(kSizesMiB[0]);
static constexpr size_t kChannelsPerConnection = 64;

struct Buffers {
  int num_gpus = 0;
  size_t elems_per_peer = 0;
  size_t bytes_per_peer = 0;
  size_t int_elems_per_peer = 0;
  size_t total_bytes = 0;
  __nv_bfloat16* input = nullptr;
  __nv_bfloat16* scratch = nullptr;
  __nv_bfloat16* output = nullptr;
  cudaStream_t stream = nullptr;
  mscclpp::RegisteredMemory registered_input;
  mscclpp::RegisteredMemory registered_scratch;
  mscclpp::RegisteredMemory registered_output;
  std::vector<mscclpp::Connection> connections;
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

__global__ void __launch_bounds__(1024) alltoall2(int rank, int nRanksPerNode, size_t nElements, void* inputBuffer,
                                                  void* scratchBuffer, void* resultBuffer) {
// TODO
}

class All2All {
 public:
  All2All(int rank, int num_gpus, const mscclpp::UniqueId& unique_id) : rank_(rank), num_gpus_(num_gpus) {
// TODO
  }

  Buffers make_buffers(size_t requested_total_bytes) {
    // TODO
  }

  void fill_inputs(Buffers& buf) const {
    // TODO
  }

  void zero_outputs(Buffers& buf) const { 
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

  void launch(Buffers& buf, bool check_launch_error = true) const {
    // TODO
  }

  void launch_rank(Buffers& buf, int rank, bool check_launch_error = true) const {
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
};

// DO NOT CHANGE CODE BEYOND THIS POINT
static void benchmark(All2All& alltoall) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int size_mib = kSizesMiB[i];
    const size_t requested_total_bytes = static_cast<size_t>(size_mib) << 20;

    Buffers buf = alltoall.make_buffers(requested_total_bytes);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    alltoall.fill_inputs(buf);
    alltoall.barrier();

    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      alltoall.zero_outputs(buf);
      alltoall.launch(buf, true);
      alltoall.sync_streams(buf);
      alltoall.barrier();
    }

    CUDA_CHECK(cudaStreamBeginCapture(buf.stream, cudaStreamCaptureModeGlobal));
    for (int iter = 0; iter < kBenchIters; ++iter) {
      alltoall.launch_rank(buf, alltoall.rank(), false);
    }
    CUDA_CHECK(cudaStreamEndCapture(buf.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    alltoall.zero_outputs(buf);
    alltoall.barrier();
    auto start = std::chrono::high_resolution_clock::now();
    for (int launch = 0; launch < kCudaGraphLaunches; ++launch) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, buf.stream));
    }
    alltoall.sync_streams(buf);
    alltoall.barrier();
    auto end = std::chrono::high_resolution_clock::now();

    const double local_total_us = std::chrono::duration<double, std::micro>(end - start).count();
    const double local_avg_us = local_total_us / static_cast<double>(kBenchIters) / static_cast<double>(kCudaGraphLaunches);
    const double avg_us = alltoall.gather_avg_latency(local_avg_us);
    const double avg_sec = avg_us / 1.0e6;
    const double gb_per_s = avg_sec > 0.0 ? (static_cast<double>(buf.total_bytes) / avg_sec / 1.0e9) : 0.0;
    const bool pass = alltoall.verify(buf);
    const bool all_pass = alltoall.all_ranks_passed(pass);
    overall_pass = overall_pass && all_pass;

    if (graph_exec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));

    if (alltoall.is_root()) {
      rows.push_back(MetricRow{size_mib, buf.total_bytes, gb_per_s, avg_us, all_pass});
    }
    alltoall.free_buffers(buf);
  }

  if (!alltoall.is_root()) return;

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
  mscclpp::UniqueId unique_id{};
};

static int run_worker(const WorkerConfig& config) {
  All2All alltoall(config.rank, config.num_gpus, config.unique_id);
  benchmark(alltoall);
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
      if (value != "alltoall2") {
        std::fprintf(stderr, "Unknown --mode '%s'; this example only supports alltoall2\n", value.c_str());
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
