/*
Intra-node local reduce-scatter (localReduceScatter): 
Iterates over peers within the node using a ring-like pattern. 
On each step, one thread (block 0, thread 0) issues a putWithSignal to send data 
to a peer and waits for data from another peer. 
After waiting, all blocks perform an in-place vectorized integer sum (vectorSum) 
on the received scratch data into the output buffer. 
The data sent is always the chunk destined for the local rank's final ownership, 
read from the source rank's slice.


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

Assumed running command:
/home/uccl/miniconda3/bin/nvcc -std=c++17 -x cu -DMSCCLPP_FORCE_DISABLE_NVLS=1 -ccbin /usr/bin/g++ --compiler-options -B/usr/bin -O3 -arch=sm_100a \
  -I /home/uccl/yuyi/llm-for-gpu-comm/datasets/example37_mscclpp_reducescatter_easy \
  -I /home/uccl/yuyi/llm-for-gpu-comm/datasets/build_mscclpp/include \
  -I /home/uccl/yuyi/llm-for-gpu-comm/datasets/third_party/mscclpp/test/mscclpp-test \
  -L /home/uccl/yuyi/llm-for-gpu-comm/datasets/build_mscclpp/lib \
  -Xlinker -rpath=/home/uccl/yuyi/llm-for-gpu-comm/datasets/build_mscclpp/lib \
  -L /home/uccl/miniconda3/targets/x86_64-linux/lib \
  -Xlinker -rpath=/home/uccl/miniconda3/targets/x86_64-linux/lib \
  -L /home/uccl/miniconda3/lib \
  -Xlinker -rpath=/home/uccl/miniconda3/lib \
  -lmscclpp -lcudart -lcuda -lnuma \
  ref_mscclpp_reducescatter.cu -o ref_mscclpp_reducescatter

./ref_mscclpp_reducescatter --gpus 8
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

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;

__constant__ DeviceHandle<mscclpp::PortChannel> constDevFstRoundChans[16];
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
static constexpr int kKernel3Blocks = 24;
static constexpr int kKernel3Threads = 1024;
static constexpr int kSizesMiB[] = {1, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536};
static constexpr int kNumSizes = sizeof(kSizesMiB) / sizeof(kSizesMiB[0]);

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

__global__ void verify_chunk_kernel(const int* ptr, size_t offset, size_t elems, int expected, int* error_count) {
// TODO
}

__forceinline__ __device__ void vectorSum(int* dst, int* src, size_t nElem, int blockId, int nBlocks) {
// TODO
}

__forceinline__ __device__ void vectorSum(int* dst, int* src, size_t nElem) {
  vectorSum(dst, src, nElem, blockIdx.x, gridDim.x);
}

__device__ void localReduceScatter(int* buff, int* scratch, int rank, int nRanksPerNode, int startChunkIndex,
                                   size_t offsetInChunk, size_t chunkSize, size_t nelems) {
                                    // TODO
}

__device__ void reduceScatter(int* buff, int* scratch, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
// TODO
}

extern "C" __global__ void __launch_bounds__(kKernel3Threads, 1)
    reducescatter_only3(int* buff, int* scratch, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  // TODO
}

class ReduceScatter {
 public:
  ReduceScatter(int rank, int num_gpus, const mscclpp::UniqueId& unique_id) : rank_(rank), num_gpus_(num_gpus) {
// TODO
  }

  Buffers make_buffers(size_t requested_total_bytes) {
// TODO
  }

  void fill_inputs(Buffers& buf) const {
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

  void launch(Buffers& buf) const {
// TODO
  }

  void sync_stream(Buffers& buf) const {
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
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap_;
  std::shared_ptr<mscclpp::Communicator> communicator_;
  // TODO: implement private functions as needed
};

static void benchmark(ReduceScatter& rs) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int size_mib = kSizesMiB[i];
    const size_t requested_total_bytes = static_cast<size_t>(size_mib) << 20;

    Buffers buf = rs.make_buffers(requested_total_bytes);

    // Correctness check on first size
    if (i == 0) {
      rs.fill_inputs(buf);
      rs.barrier();
      rs.launch(buf);
      rs.sync_stream(buf);
      rs.barrier();
      const bool pass = rs.verify(buf);
      const bool all_pass = rs.all_ranks_passed(pass);
      if (rs.is_root()) {
        std::fprintf(stderr, "reduce-scatter correctness check: %s\n", all_pass ? "PASS" : "FAIL");
      }
    }

    // Warmup
    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      rs.fill_inputs(buf);
      rs.barrier();
      rs.launch(buf);
      rs.sync_stream(buf);
      rs.barrier();
    }

    // Benchmark
    rs.fill_inputs(buf);
    rs.barrier();
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < kBenchIters; ++iter) {
      rs.fill_inputs(buf);
      rs.barrier();
      rs.launch(buf);
      rs.sync_stream(buf);
      rs.barrier();
    }
    auto end = std::chrono::high_resolution_clock::now();

    const double local_total_us = std::chrono::duration<double, std::micro>(end - start).count();
    const double local_avg_us = local_total_us / static_cast<double>(kBenchIters);
    const double avg_us = rs.gather_avg_latency(local_avg_us);
    const double avg_sec = avg_us / 1.0e6;
    const double gb_per_s = avg_sec > 0.0 ? (static_cast<double>(buf.total_bytes) / avg_sec / 1.0e9) : 0.0;
    const bool pass = rs.verify(buf);
    const bool all_pass = rs.all_ranks_passed(pass);
    overall_pass = overall_pass && all_pass;

    if (rs.is_root()) {
      rows.push_back(MetricRow{size_mib, buf.total_bytes, gb_per_s, avg_us, all_pass});
    }
    rs.free_buffers(buf);
  }

  if (!rs.is_root()) return;

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
  ReduceScatter rs(config.rank, config.num_gpus, config.unique_id);
  benchmark(rs);
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
    if (arg == "--gpus" && i + 1 < argc) {
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