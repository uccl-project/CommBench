// Standalone launcher around SGLang's fused tensor-parallel QK RMSNorm kernel.
//
// In tensor-parallel LLM inference, QK-norm (RMSNorm applied to Q and K before attention) 
// requires a global sum of squares across all TP ranks before any rank can compute its local scale factor. 
// Naively this is a two-kernel sequence: AllReduce the partial sums, 
// then compute the norm. This kernel fuses both into a single-kernel, 
// push-based custom allreduce + RMSNorm, 
// running entirely on the GPU without touching NCCL.
//
// Example from repo root:
//  nvcc --std=c++20 --expt-relaxed-constexpr -x cu -O3 -arch=sm_100a \
  -DSGL_CUDA_ARCH=1000 \
  -I /home/uccl/yuyi/llm-for-gpu-comm/datasets/third_party/sglang/python/sglang/jit_kernel/include \
  -I $(python3 -c 'import pathlib, tvm_ffi; print(pathlib.Path(tvm_ffi.__file__).parent / "include")') \
  ref_sglang_qknorm_fused.cu -o ref_sglang_qknorm_fused

//
//   ./ref_sglang_qknorm_fused --gpus 1 or 2 or 4 or 8

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <sgl_kernel/distributed/common.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

using device::distributed::PushController;

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
static constexpr int kGraphKernelIters = 1;
static constexpr int kGraphReplayIters = 100;
static constexpr int kQDim = 6144;
static constexpr int kKDim = 1024;
static constexpr float kEps = 1.0e-6f;
static constexpr uint32_t kMaxPushBlocks = 4096;
static constexpr int kTokenSizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
static constexpr int kNumSizes = sizeof(kTokenSizes) / sizeof(kTokenSizes[0]);

struct RankBuffers {
  uint32_t num_tokens = 0;
  int64_t local_q_dim = 0;
  int64_t local_k_dim = 0;
  size_t q_elems = 0;
  size_t k_elems = 0;
  size_t q_bytes = 0;
  size_t k_bytes = 0;
  size_t weight_q_bytes = 0;
  size_t weight_k_bytes = 0;
  __nv_bfloat16* q = nullptr;
  __nv_bfloat16* k = nullptr;
  __nv_bfloat16* q_weight = nullptr;
  __nv_bfloat16* k_weight = nullptr;
  cudaStream_t stream = nullptr;
  std::vector<__nv_bfloat16> host_q;
  std::vector<__nv_bfloat16> host_k;
  std::vector<__nv_bfloat16> host_q_weight;
  std::vector<__nv_bfloat16> host_k_weight;
};

struct RankState {
  uint32_t* push_signal = nullptr;
  void* push_buffer = nullptr;
  std::vector<void*> peer_push_buffers;
};

struct MetricRow {
  int tokens;
  size_t actual_bytes;
  double throughput_avg_gb_per_s;
  double latency_avg_us;
  bool pass;
};

struct GraphSet {
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> execs;
};

namespace {

size_t align_bytes(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

float q_value(int rank, int token, int local_idx) {
  return 0.01f * static_cast<float>((rank + 1) * 3 + (token % 7)) +
         0.001f * static_cast<float>((local_idx % 17) - 8);
}

float k_value(int rank, int token, int local_idx) {
  return 0.02f * static_cast<float>((rank + 1) * 2 + (token % 5)) -
         0.0015f * static_cast<float>((local_idx % 13) - 6);
}

float weight_value(int global_idx) {
  return 0.75f + 0.01f * static_cast<float>(global_idx % 31);
}

void enable_peer_access(int num_gpus) {
  for (int rank = 0; rank < num_gpus; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    for (int peer = 0; peer < num_gpus; ++peer) {
      if (peer == rank) continue;
      int can_access = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, rank, peer));
      if (!can_access) {
        throw std::runtime_error("CUDA peer access is required for this standalone SGLang qknorm test");
      }
      cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else {
        CUDA_CHECK(err);
      }
    }
  }
}

}  // namespace

struct ParallelQKNormParams {
  void* __restrict__ buffer[device::distributed::kMaxNumGPU];
  void* q_ptr;
  void* k_ptr;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  float eps;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t epoch_bytes;
  uint32_t num_clean_up_count = 0;
};

template <typename T>
SGL_DEVICE void ld_global_volatile_8B(T& x, const void* addr, int64_t offset) {
  static_assert(alignof(T) == 8 && sizeof(T) == 8);
  addr = device::pointer::offset<T>(addr, offset);
  uint2 val;
  asm volatile("ld.volatile.global.v2.b32 {%0, %1}, [%2];" : "=r"(val.x), "=r"(val.y) : "l"(addr));
  x = *reinterpret_cast<const T*>(&val);
}

template <typename T>
SGL_DEVICE void st_global_volatile_8B(const T& x, void* addr, int64_t offset) {
  static_assert(alignof(T) == 8 && sizeof(T) == 8);
  const uint2 val = *reinterpret_cast<const uint2*>(&x);
  addr = device::pointer::offset<T>(addr, offset);
  asm volatile("st.volatile.global.v2.b32 [%2], {%0, %1};" ::"r"(val.x), "r"(val.y), "l"(addr));
}

SGL_DEVICE float sync_float(float x) {
  return __shfl_sync(0xffffffffu, x, 0);
}

constexpr uint32_t next_pow_of_2(uint32_t x) {
  uint32_t y = 1;
  while (y < x) y *= 2;
  return y;
}

template <typename DType_, uint32_t kNumGPU_, int64_t kQDim_, int64_t kKDim_, bool kUsePDL_>
struct KernelTrait {
  using DType = DType_;
  static constexpr uint32_t kNumGPU = kNumGPU_;
  static constexpr int64_t kQDim = kQDim_;
  static constexpr int64_t kKDim = kKDim_;
  static constexpr bool kUsePDL = kUsePDL_;

  static constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  static constexpr int64_t kLocalQDim = kQDim / kNumGPU;
  static constexpr int64_t kLocalKDim = kKDim / kNumGPU;
  static constexpr uint32_t kNumQThreads = kLocalQDim / (kVecSize * 2);
  static constexpr uint32_t kNumKThreads = kLocalKDim / (kVecSize * 2);
  static constexpr uint32_t kNumQWarps = kNumQThreads / device::kWarpThreads;
  static constexpr uint32_t kNumKWarps = host::div_ceil(kNumKThreads, device::kWarpThreads);
  static constexpr uint32_t kBlockSize = (kNumQWarps + kNumKWarps) * device::kWarpThreads;
  static constexpr uint32_t kOccupancy = 2048 / kBlockSize;

  using DType2 = packed_t<DType>;
  using Storage = device::AlignedVector<DType2, kVecSize>;

  static_assert(kNumGPU != 0 && (kNumGPU & (kNumGPU - 1)) == 0, "must be pow of 2");
  static_assert(kQDim % kNumGPU == 0);
  static_assert(kKDim % kNumGPU == 0);
  static_assert(kLocalQDim % (kVecSize * 2) == 0);
  static_assert(kLocalKDim % (kVecSize * 2) == 0);
  static_assert(kNumQThreads % device::kWarpThreads == 0);
  static_assert(kBlockSize <= 1024);
  static_assert(sizeof(Storage) == 16 && alignof(Storage) == 16);
  static_assert(kOccupancy * kBlockSize <= 2048);
};

template <typename Trait>
__global__ __launch_bounds__(Trait::kBlockSize, Trait::kOccupancy) void parallel_qknorm_across_head(
    const ParallelQKNormParams __grid_constant__ params, const PushController __grid_constant__ ctrl) {
// TODO
}

class QKNormFusedStandalone {
 public:
  explicit QKNormFusedStandalone(int num_gpus) : num_gpus_(num_gpus), states_(num_gpus) {
// TODO
  }

  ~QKNormFusedStandalone() {
// TODO
  }

  std::vector<RankBuffers> make_buffers(uint32_t num_tokens) const {
// TODO
  }

  void fill_inputs(std::vector<RankBuffers>& bufs) const {
// TODO
  }

  void launch(std::vector<RankBuffers>& bufs, bool check_launch_error = true) const {
// TODO
  }

  void sync_streams(std::vector<RankBuffers>& bufs) const {
// TODO
  }

  bool verify(std::vector<RankBuffers>& bufs) const {
// TODO
  }

  void free_buffers(std::vector<RankBuffers>& bufs) const {
// TODO
  }

  GraphSet capture_graph(std::vector<RankBuffers>& bufs, int launches_per_graph) const {
// TODO
  }

  void launch_graph(const GraphSet& graph_set, std::vector<RankBuffers>& bufs) const {
// TODO
  }

  void destroy_graph(GraphSet& graph_set) const {
// TODO
  }
};

// DO NOT CHANGE CODE BEYOND THIS POINT

static void benchmark(QKNormFusedStandalone& qknorm) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int tokens = kTokenSizes[i];
    std::vector<RankBuffers> bufs = qknorm.make_buffers(tokens);
    qknorm.fill_inputs(bufs);
    GraphSet graph_set = qknorm.capture_graph(bufs, kGraphKernelIters);

    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      qknorm.launch_graph(graph_set, bufs);
    }
    qknorm.sync_streams(bufs);

    std::vector<cudaEvent_t> start_events(bufs.size());
    std::vector<cudaEvent_t> end_events(bufs.size());
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventCreate(&start_events[rank]));
      CUDA_CHECK(cudaEventCreate(&end_events[rank]));
      CUDA_CHECK(cudaEventRecord(start_events[rank], bufs[rank].stream));
    }
    for (int iter = 0; iter < kGraphReplayIters; ++iter) {
      qknorm.launch_graph(graph_set, bufs);
    }
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventRecord(end_events[rank], bufs[rank].stream));
    }
    qknorm.sync_streams(bufs);

    double avg_us = 0.0;
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_events[rank], end_events[rank]));
      avg_us += static_cast<double>(elapsed_ms) * 1000.0 /
                static_cast<double>(kGraphKernelIters * kGraphReplayIters);
      CUDA_CHECK(cudaEventDestroy(start_events[rank]));
      CUDA_CHECK(cudaEventDestroy(end_events[rank]));
    }
    avg_us /= static_cast<double>(bufs.size());

    qknorm.fill_inputs(bufs);
    qknorm.launch(bufs, true);
    qknorm.sync_streams(bufs);

    const size_t actual_bytes =
        (bufs[0].q_bytes + bufs[0].k_bytes + bufs[0].weight_q_bytes + bufs[0].weight_k_bytes) * bufs.size();
    const double avg_sec = avg_us / 1.0e6;
    const double gb_per_s = avg_sec > 0.0 ? (static_cast<double>(actual_bytes) / avg_sec / 1.0e9) : 0.0;
    const bool pass = qknorm.verify(bufs);
    overall_pass = overall_pass && pass;
    rows.push_back(MetricRow{tokens, actual_bytes, gb_per_s, avg_us, pass});
    qknorm.destroy_graph(graph_set);
    qknorm.free_buffers(bufs);
  }

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", overall_pass ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"tokens\",\n");
  std::printf("  \"actual_bytes_unit\": \"B\",\n");
  std::printf("  \"throughput_unit\": \"GB/s\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < rows.size(); ++i) {
    const MetricRow& row = rows[i];
    std::printf("    {\"data_size\": %d, \"actual_bytes\": %zu, \"throughput_avg\": %.3f, "
                "\"latency_avg\": %.3f, \"pass\": %s}",
                row.tokens, row.actual_bytes, row.throughput_avg_gb_per_s, row.latency_avg_us,
                row.pass ? "true" : "false");
    if (i + 1 != rows.size()) std::printf(",");
    std::printf("\n");
  }
  std::printf("  ]\n");
  std::printf("}\n");
}

int main(int argc, char** argv) {
  int requested_gpus = kMaxGpus;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      std::string value = argv[++i];
      if (value != "qknorm") {
        std::fprintf(stderr, "Unknown --mode '%s'; this example only supports qknorm\n", value.c_str());
        return EXIT_FAILURE;
      }
    } else if (arg == "--gpus" && i + 1 < argc) {
      requested_gpus = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "Usage: %s [--gpus N] [--mode qknorm]\n", argv[0]);
      return EXIT_FAILURE;
    }
  }

  int available_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
  const int num_gpus = std::min({requested_gpus, available_gpus, kMaxGpus});
  if (!(num_gpus == 1 || num_gpus == 2 || num_gpus == 4 || num_gpus == 8)) {
    std::fprintf(stderr, "This standalone qknorm test supports 1, 2, 4, or 8 GPUs; got %d\n", num_gpus);
    return EXIT_FAILURE;
  }

  QKNormFusedStandalone qknorm(num_gpus);
  benchmark(qknorm);
  return EXIT_SUCCESS;
}
