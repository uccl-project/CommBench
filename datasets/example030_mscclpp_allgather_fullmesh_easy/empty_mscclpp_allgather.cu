// Standalone CUDA allgather derived from the MSCCl++ allgather6 test kernel.
//
// You are an expert in GPU Comm kernel and optimization
// Your task is to implement most performant allgather kernel and related host code in the AllGather class according to the TODO comments. 
// Feel free to reference any MSCCLPP based allgather implementations and optimizations, but this file needs to be standalone.
// The main function and benchmark logic are provided for you. 
// You can add helper functions or members to the AllGather class if needed, for example to manage CUDA graphs or other resources.
// The testing will first be on: correctness, and then performance (throughput and latency) of your implementation. 
// The benchmark will run multiple iterations for different data sizes and report the average throughput and latency.
//
// Hardware topology assumptions:
// 8 B300 GPUs. 
// Compilation assumptions
//   nvcc --std=c++17 -x cu -O3 -arch=sm_100a \
//     -I /workspace/llm-for-gpu-comm/datasets/example30_mscclpp_allgather_fullmesh \
//     -I /workspace/llm-for-gpu-comm/datasets/build_mscclpp/include \
//     -I /workspace/llm-for-gpu-comm/datasets/third_party/mscclpp/test/mscclpp-test \
//     -L /workspace/llm-for-gpu-comm/datasets/build_mscclpp/lib \
//     -Xlinker -rpath=/workspace/llm-for-gpu-comm/datasets/build_mscclpp/lib \
//     -lmscclpp \
//     ref_mscclpp_allgather.cu -o ref_mscclpp_allgather
//
//   ./ref_mscclpp_allgather --gpus 1 or 2 or 8

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
#include <mscclpp/gpu_utils.hpp>
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
static constexpr int kKernel6Blocks = 24;
static constexpr int kKernel6Threads = 1024;
static constexpr int kSizesMiB[] = {1, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536};
static constexpr int kNumSizes = sizeof(kSizesMiB) / sizeof(kSizesMiB[0]);
static constexpr size_t kChannelsPerConnection = 64;

struct Buffers {
  int num_gpus = 0;
  size_t elems_per_gpu = 0;
  size_t bytes_per_gpu = 0;
  size_t total_bytes = 0;
  std::shared_ptr<__nv_bfloat16> owned_output;
  __nv_bfloat16* output = nullptr;
  cudaStream_t stream = nullptr;
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

__global__ void __launch_bounds__(1024, 1)
    kernel6(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
      // TODO
      // this is the device side GPU Comm kernel
      // Each GPU runs this kernel simultaneously. 
      // Before doing any data movement, all peers do a lightweight barrier — each warp that "owns" a peer signals it and waits for its signal back, 
      // ensuring everyone's input buffer is ready before any pushing begins.
      // Then each GPU pushes its own local data shard to every other GPU on the node. 
      // The work is distributed across warps in a round-robin fashion — warp 0 handles peer 0, warp 1 handles peer 1, ..., warp N handles peer 0 again, and so on. 
      // This way all peers receive data concurrently rather than sequentially.
      // The data is transferred in fixed-size chunks using put(), which writes directly into the remote GPU's buffer at the correct offset 
      // (the sending rank's slot in the destination's allgather output buffer).
      //  The chunk size adapts based on total data volume — larger chunks (64 bytes/thread) when there's enough work to fill them, 
      // smaller (16 bytes/thread) otherwise. 
      // A final cleanup pass handles any leftover bytes that don't fill a complete chunk.
      // the final objective is: When all GPUs finish running this kernel, 
      // every GPU holds the complete allgathered result — 
      // each GPU's shard has been written into the correct slot of every other GPU's output buffer by the owning GPU itself.
}

class AllGather {
 public:
  AllGather(int rank, int num_gpus, const mscclpp::UniqueId& unique_id) : rank_(rank), num_gpus_(num_gpus) {
    // todo
  }

  Buffers make_buffers(size_t requested_total_bytes) {
    // todo
  }

  void fill_inputs(Buffers& buf) const {
    // todo
  }

  void zero_outputs(Buffers& buf) const { fill_inputs(buf); }

  bool verify(Buffers& buf) const {
    // todo
  }

  bool all_ranks_passed(bool local_pass) const {
    // todo
  }

  double gather_avg_latency(double local_avg_us) const {
    std::vector<double> gathered(num_gpus_, 0.0);
    gathered[rank_] = local_avg_us;
    bootstrap_->allGather(gathered.data(), sizeof(double));
    double sum = 0.0;
    for (double v : gathered) sum += v;
    return sum / static_cast<double>(gathered.size());
  }

  void launch(Buffers& buf, bool check_launch_error = true) const {
    // todo
  }

  void launch_rank(Buffers& buf, int rank, bool check_launch_error = true) const {
    // todo
  }

  void sync_streams(Buffers& buf) const {
    // todo
  }

  void free_buffers(Buffers& buf) const {
    // todo
  }

  void barrier() const { 
    // todo 
    }
  int rank() const { 
    // todo 
    }
  bool is_root() const { 
    // todo
  }

 private:
  int rank_;
  int num_gpus_;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap_;
  std::shared_ptr<mscclpp::Communicator> communicator_;
};


// do not change code beyond this line!

static void benchmark(AllGather& allgather) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int size_mib = kSizesMiB[i];
    const size_t requested_total_bytes = static_cast<size_t>(size_mib) << 20;

    Buffers buf = allgather.make_buffers(requested_total_bytes);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    allgather.fill_inputs(buf);
    allgather.barrier();

    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      allgather.zero_outputs(buf);
      allgather.launch(buf, true);
      allgather.sync_streams(buf);
      allgather.barrier();
    }

    CUDA_CHECK(cudaStreamBeginCapture(buf.stream, cudaStreamCaptureModeGlobal));
    for (int iter = 0; iter < kBenchIters; ++iter) {
      allgather.launch_rank(buf, allgather.rank(), false);
    }
    CUDA_CHECK(cudaStreamEndCapture(buf.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    allgather.zero_outputs(buf);
    allgather.barrier();
    auto start = std::chrono::high_resolution_clock::now();
    for (int launch = 0; launch < kCudaGraphLaunches; ++launch) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, buf.stream));
    }
    allgather.sync_streams(buf);
    allgather.barrier();
    auto end = std::chrono::high_resolution_clock::now();

    const double local_total_us = std::chrono::duration<double, std::micro>(end - start).count();
    const double local_avg_us = local_total_us / static_cast<double>(kBenchIters) / static_cast<double>(kCudaGraphLaunches);
    const double avg_us = allgather.gather_avg_latency(local_avg_us);
    const double avg_sec = avg_us / 1.0e6;
    const double gb_per_s = avg_sec > 0.0 ? (static_cast<double>(buf.total_bytes) / avg_sec / 1.0e9) : 0.0;
    const bool pass = allgather.verify(buf);
    const bool all_pass = allgather.all_ranks_passed(pass);
    overall_pass = overall_pass && all_pass;

    if (graph_exec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));

    if (allgather.is_root()) {
      rows.push_back(MetricRow{size_mib, buf.total_bytes, gb_per_s, avg_us, all_pass});
    }
    allgather.free_buffers(buf);
  }

  if (!allgather.is_root()) return;

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
  AllGather allgather(config.rank, config.num_gpus, config.unique_id);
  benchmark(allgather);
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
      if (value != "kernel6") {
        std::fprintf(stderr, "Unknown --mode '%s'; this example only supports kernel6\n", value.c_str());
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
