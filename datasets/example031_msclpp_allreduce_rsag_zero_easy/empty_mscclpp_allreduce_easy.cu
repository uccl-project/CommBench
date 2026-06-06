/*
Implement a high-performance AllReduce collective communication kernel using the MSCCL++ library 
targeting a single NVLink-connected node of up to 8 GPUs. 
The implementation uses a reduce-scatter then all-gather decomposition strategy.

The reduce-scatter uses a pipelined, two-level approach:

Intra-node local reduce-scatter (localReduceScatter): 
Iterates over peers within the node using a ring-like pattern. 
On each step, one thread (block 0, thread 0) issues a putWithSignal to send data 
to a peer and waits for data from another peer. 
After waiting, all blocks perform an in-place vectorized integer sum (vectorSum) 
on the received scratch data into the output buffer. 
The data sent is always the chunk destined for the local rank's final ownership, 
read from the source rank's slice.

Inter-node port channel exchange (for multi-node generalization stubs, active even in single-node via the peer node logic): 
Overlaps communication and computation using a pipeline of depth 3 (pipelineSize = 3). 
The first pipeline stage sends chunkSize / 3 elements while local reduce-scatter 
runs on the remaining 2 * chunkSize / 3. The second stage sends the remainder while 
the local reduce-scatter finishes, then reduces the received partial results with vectorSum.

All-Gather Phase
The all-gather uses memory channels for direct peer writes:
Block 0 performs a handshake with all peers using relaxedSignal + wait before the actual data movement begins.
After a grid-wide sync, all blocks participate in the data transfer. 
Each block accesses its own channel slice (constMemChans + nPeer * blockIdx.x).
Data is distributed across warps: each warp writes unitBytesPerWarp bytes 
(either 64 or 16 bytes per thread depending on total data size) 
to a peer's buffer at the correct offset for the local rank's chunk. 
The work is distributed round-robin across peers by warp index.
A final partial-remainder pass handles bytes that don't evenly divide across warps, using put<16, true> (bounds-checked variant).

Multi-process execution model: 
Each GPU rank runs in a separate process (spawned via fork + execl), not a thread. 
The bootstrap unique ID is passed via CLI argument as a hex string.

__constant__ memory indexing contract: 
The memory channel layout (nPeer * blockIdx.x stride) must exactly match how channels were registered
This is a correctness-critical invariant.

putWithSignal is single-threaded: 
Only thread 0 of block 0 (isComm) issues port channel operations. 
All other threads participate only in vectorSum and deviceSyncer.sync.

Channel count constraint: 
constMemChans is statically sized at 512 entries; 
with 8 GPUs and 64 channels per connection, that's exactly 7 * 64 = 448 channels

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
#include <unordered_map>
#include <vector>

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;

__constant__ DeviceHandle<mscclpp::PortChannel> constDevFstRoundChans[16];
__constant__ DeviceHandle<mscclpp::PortChannel> constDevSndRoundChans[16];
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemChans[512];
__device__ mscclpp::DeviceSyncer deviceSyncer;

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
static constexpr int kKernel3Blocks = 24;
static constexpr int kKernel3Threads = 1024;
static constexpr int kSizesMiB[] = {1, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536};
static constexpr int kNumSizes = sizeof(kSizesMiB) / sizeof(kSizesMiB[0]);
static constexpr size_t kChannelsPerConnection = 64;

struct Buffers {
  int num_gpus = 0;
  size_t elems_per_gpu = 0;
  size_t bytes_per_gpu = 0;
  size_t total_bytes = 0;
  std::shared_ptr<int> owned_output;
  std::shared_ptr<int> owned_scratch;
  int* output = nullptr;
  int* scratch = nullptr;
  cudaStream_t stream = nullptr;
  mscclpp::RegisteredMemory registered_output;
  mscclpp::RegisteredMemory registered_scratch;
  std::vector<mscclpp::Connection> fst_round_connections;
  std::vector<mscclpp::Connection> snd_round_connections;
  std::vector<mscclpp::Connection> memory_connections;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> semaphores;
  std::vector<mscclpp::MemoryChannel> memory_channels;
  std::shared_ptr<mscclpp::BaseProxyService> proxy_service;
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

__global__ void fill_buffer_kernel(int* ptr, size_t elems, int value) {
// TODO
}

__global__ void fill_own_chunk_kernel(int* ptr, size_t elems_per_gpu, int rank, int value) {
// TODO
}

__global__ void verify_allreduce_kernel(const int* ptr, size_t elems, int expected, int* error_count) {
// TODO
}

__global__ void verify_chunk_kernel(const int* ptr, size_t offset, size_t elems, int expected, int* error_count) {
// TODO
}

__forceinline__ __device__ void vectorSum(int* dst, int* src, size_t nElem, int blockId, int nBlocks) {
  // TODO
}

__forceinline__ __device__ void vectorSum(int* dst, int* src, size_t nElem) {
  // TODO
}

__device__ void localReduceScatter(int* buff, int* scratch, int rank, int nRanksPerNode, int startChunkIndex,
                                   size_t offsetInChunk, size_t chunkSize, size_t nelems) {
    // TODO
}

__device__ void reduceScatter(int* buff, int* scratch, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
// TODO
}

// Run with a single thread only.
__device__ void localAllGather(int rank, int nRanksPerNode, uint64_t offset, uint64_t size) {
// TODO
}

// Run with a single thread only.
__device__ void allGather(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
// TODO
}


extern "C" __global__ void __launch_bounds__(kKernel3Threads, 1)
// TODO
}

extern "C" __global__ void __launch_bounds__(kKernel3Threads, 1)
// TODO
}

extern "C" __global__ void __launch_bounds__(kKernel3Threads, 1)
// TODO
}

class AllReduce {
 public:
  AllReduce(int rank, int num_gpus, const mscclpp::UniqueId& unique_id) : rank_(rank), num_gpus_(num_gpus) {
// TODO
  }

  Buffers make_buffers(size_t requested_total_bytes) {
// TODO
  }

  void fill_inputs(Buffers& buf) const {
// TODO
  }

  void zero_outputs(Buffers& buf) const { fill_inputs(buf); }

  bool verify(Buffers& buf) const {
// TODO
  }

  bool verify_reducescatter(Buffers& buf) const {
// TODO
  }

  bool verify_allgather_pattern(Buffers& buf) const {
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

  void launch_reducescatter_only(Buffers& buf, bool check_launch_error = true) const {
// TODO
  }

  void launch_allgather_only(Buffers& buf, bool check_launch_error = true) const {
// TODO
  }

  void prepare_allgather_only_inputs(Buffers& buf) const {
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

  void barrier() const { // TODO 
    }
  int rank() const { // TODO 
    }
  bool is_root() const { // TODO 
    }

 private:
  int rank_;
  int num_gpus_;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap_;
  std::shared_ptr<mscclpp::Communicator> communicator_;

  void setup_mesh_connections_internal(std::vector<mscclpp::Connection>& connections,
                                       mscclpp::RegisteredMemory& local_memory,
                                       std::vector<std::shared_future<mscclpp::RegisteredMemory>>& remote_memories) {
// TODO
  }

  void setup_port_channels(std::vector<DeviceHandle<mscclpp::PortChannel>>& port_channels,
                           std::vector<mscclpp::Connection>& connections, mscclpp::RegisteredMemory& local_memory,
                           mscclpp::RegisteredMemory& input_memory,
                           const std::shared_ptr<mscclpp::ProxyService>& service) {
// TODO
  }

  void setup_memory_channels(Buffers& buf) {
// TODO
  }

  void setup_connections(Buffers& buf) {
// TODO
  }
};

static void benchmark(AllReduce& allreduce) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int size_mib = kSizesMiB[i];
    const size_t requested_total_bytes = static_cast<size_t>(size_mib) << 20;

    Buffers buf = allreduce.make_buffers(requested_total_bytes);

    allreduce.fill_inputs(buf);
    allreduce.barrier();

    // Debug-only checks (reduce-scatter-only / allgather-only / single-launch)
    // intentionally disabled for normal benchmark runs.
    if (i == 0) {
      allreduce.zero_outputs(buf);
      allreduce.barrier();
      allreduce.launch_reducescatter_only(buf, true);
      allreduce.sync_streams(buf);
      allreduce.barrier();
      const bool rs_pass = allreduce.verify_reducescatter(buf);
      const bool rs_all_pass = allreduce.all_ranks_passed(rs_pass);
      if (allreduce.is_root()) {
        std::fprintf(stderr, "reduce-scatter-only check: %s\n", rs_all_pass ? "PASS" : "FAIL");
      }
      allreduce.zero_outputs(buf);
      allreduce.barrier();

      allreduce.prepare_allgather_only_inputs(buf);
      allreduce.barrier();
      allreduce.launch_allgather_only(buf, true);
      allreduce.sync_streams(buf);
      allreduce.barrier();
      const bool ag_pass = allreduce.verify_allgather_pattern(buf);
      const bool ag_all_pass = allreduce.all_ranks_passed(ag_pass);
      if (allreduce.is_root()) {
        std::fprintf(stderr, "allgather-only check: %s\n", ag_all_pass ? "PASS" : "FAIL");
      }
      allreduce.zero_outputs(buf);
      allreduce.barrier();

      allreduce.zero_outputs(buf);
      allreduce.barrier();
      allreduce.launch(buf, true);
      allreduce.sync_streams(buf);
      allreduce.barrier();
      const bool single_pass = allreduce.verify(buf);
      const bool single_all_pass = allreduce.all_ranks_passed(single_pass);
      if (allreduce.is_root()) {
        std::fprintf(stderr, "single-launch allreduce check: %s\n", single_all_pass ? "PASS" : "FAIL");
      }
      allreduce.zero_outputs(buf);
      allreduce.barrier();
    }

    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      allreduce.zero_outputs(buf);
      allreduce.barrier();
      allreduce.launch(buf, true);
      allreduce.sync_streams(buf);
      allreduce.barrier();
    }

    double local_total_us = 0.0;
    for (int launch = 0; launch < kCudaGraphLaunches; ++launch) {
      for (int iter = 0; iter < kBenchIters; ++iter) {
        allreduce.zero_outputs(buf);
        allreduce.barrier();
        auto start = std::chrono::high_resolution_clock::now();
        allreduce.launch_rank(buf, allreduce.rank(), true);
        allreduce.sync_streams(buf);
        auto end = std::chrono::high_resolution_clock::now();
        local_total_us += std::chrono::duration<double, std::micro>(end - start).count();
        allreduce.barrier();
      }
    }

    const double local_avg_us =
        local_total_us / static_cast<double>(kBenchIters) / static_cast<double>(kCudaGraphLaunches);
    const double avg_us = allreduce.gather_avg_latency(local_avg_us);
    const double avg_sec = avg_us / 1.0e6;
    const double gb_per_s = avg_sec > 0.0 ? (static_cast<double>(buf.total_bytes) / avg_sec / 1.0e9) : 0.0;
    const bool pass = allreduce.verify(buf);
    const bool all_pass = allreduce.all_ranks_passed(pass);
    overall_pass = overall_pass && all_pass;

    if (allreduce.is_root()) {
      rows.push_back(MetricRow{size_mib, buf.total_bytes, gb_per_s, avg_us, all_pass});
    }
    allreduce.free_buffers(buf);
  }

  if (!allreduce.is_root()) return;

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
  AllReduce allreduce(config.rank, config.num_gpus, config.unique_id);
  benchmark(allreduce);
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
