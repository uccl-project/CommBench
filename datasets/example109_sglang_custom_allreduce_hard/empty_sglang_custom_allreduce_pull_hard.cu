// Custom intra-process all-reduce across multiple CUDA devices.
//
// You are given only the test harness contract and the JSON output
// scaffolding below.  The template intentionally provides no
// implementation hints — no SGLang headers, no helper kernels, no
// device-side helpers, no peer-access setup.  Implement everything
// from scratch: data buffers, peer access, the one-shot pull kernel,
// correctness verification, and resource teardown.
//
// Everything above `// DO NOT CHANGE CODE BEYOND THIS POINT` is yours
// to extend or rewrite, as long as the names and signatures referenced
// by the harness keep working:
//
//   struct AllReduceData             struct AllReduceParams
//   struct RankBuffers               struct MetricRow
//   class  CustomAllReducePullStandalone
//     ctor(int num_gpus)
//     std::vector<RankBuffers> make_buffers(uint32_t num_items) const
//     void  fill_inputs   (std::vector<RankBuffers>&)
//     void  launch        (std::vector<RankBuffers>&, bool check_launch_error=true)
//     void  sync_streams  (std::vector<RankBuffers>&)
//     bool  verify        (std::vector<RankBuffers>&)
//     void  free_buffers  (std::vector<RankBuffers>&)
//
// The harness only reads `RankBuffers::stream` plus `sizeof(bf16_t)`,
// so feel free to add whatever extra fields/methods/types/kernels you
// need for your design.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using bf16_t = __nv_bfloat16;

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
static constexpr int kBenchmarkIters = 100;
static constexpr uint32_t kCtaSize = 256;
static constexpr int kMessageSizesBytes[] = {
    4 * 1024,       16 * 1024,      64 * 1024,      128 * 1024,
    256 * 1024,     512 * 1024,     1024 * 1024,    2 * 1024 * 1024,
    4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024};
static constexpr int kNumSizes = sizeof(kMessageSizesBytes) / sizeof(kMessageSizesBytes[0]);

struct AllReduceData {
  void* __restrict__ input[kMaxGpus];
};

struct AllReduceParams {
  void* __restrict__ output;
  uint32_t rank;
  uint32_t num_items;
};

struct RankBuffers {
  uint32_t num_items = 0;
  cudaStream_t stream = nullptr;
  // TODO: add fields for your device-side input buffer, peer-visible
  //       pull buffer, signal/semaphore primitive, host scratch, etc.
};

struct MetricRow {
  int bytes;
  size_t actual_bytes;
  double throughput_avg_gb_per_s;
  double latency_avg_us;
  bool pass;
};

class CustomAllReducePullStandalone {
 public:
  explicit CustomAllReducePullStandalone(int num_gpus) {
    // TODO: store num_gpus, enable peer access, allocate any
    //       persistent state your design needs.
  }

  std::vector<RankBuffers> make_buffers(uint32_t num_items) const {
    // TODO: per-rank buffer + stream allocation, peer registration.
    return {};
  }

  void fill_inputs(std::vector<RankBuffers>& bufs) const {
    // TODO: populate each rank's input with a reproducible pattern
    //       so verify() can check the expected reduction value.
  }

  void launch(std::vector<RankBuffers>& bufs, bool check_launch_error = true) const {
    // TODO: launch your all-reduce kernel on each rank's stream.
  }

  void sync_streams(std::vector<RankBuffers>& bufs) const {
    // TODO: synchronize every rank's stream so the host can observe
    //       the post-kernel device state.
  }

  bool verify(std::vector<RankBuffers>& bufs) const {
    // TODO: copy each rank's output back to host and check it equals
    //       the expected reduction (sum across ranks).  Return false
    //       on mismatch.
    return false;
  }

  void free_buffers(std::vector<RankBuffers>& bufs) const {
    // TODO: free device buffers, destroy streams, drop peer access.
  }
};

//DO NOT CHANGE CODE BEYOND THIS POINT
static void benchmark(CustomAllReducePullStandalone& custom_allreduce) {
  std::vector<MetricRow> rows;
  bool overall_pass = true;

  for (int i = 0; i < kNumSizes; ++i) {
    const int bytes = kMessageSizesBytes[i];
    const uint32_t num_items = static_cast<uint32_t>(bytes / sizeof(bf16_t));
    std::vector<RankBuffers> bufs = custom_allreduce.make_buffers(num_items);
    custom_allreduce.fill_inputs(bufs);

    // Eager warmup to JIT/load kernels and prime peer mappings before capture.
    for (int warmup = 0; warmup < kWarmupIters; ++warmup) {
      custom_allreduce.launch(bufs);
    }
    custom_allreduce.sync_streams(bufs);

    // Capture per-rank CUDA graphs. Stream capture is per-stream, so beginning
    // capture on every rank's stream lets a single host-side dispatch loop
    // append work to each rank's graph independently. Replaying the graph
    // amortizes host-side launch overhead across all `kBenchmarkIters` iters,
    // matching how SGLang's Python harness times this kernel.
    std::vector<cudaGraph_t> graphs(bufs.size(), nullptr);
    std::vector<cudaGraphExec_t> graph_execs(bufs.size(), nullptr);
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaStreamBeginCapture(bufs[rank].stream, cudaStreamCaptureModeThreadLocal));
    }
    for (int iter = 0; iter < kBenchmarkIters; ++iter) {
      custom_allreduce.launch(bufs, /*check_launch_error=*/false);
    }
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaStreamEndCapture(bufs[rank].stream, &graphs[rank]));
    }
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaGraphInstantiate(&graph_execs[rank], graphs[rank], nullptr, nullptr, 0));
    }

    // Replay-warmup so any lazy graph setup costs do not pollute timing.
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaGraphLaunch(graph_execs[rank], bufs[rank].stream));
    }
    custom_allreduce.sync_streams(bufs);

    std::vector<cudaEvent_t> start_events(bufs.size());
    std::vector<cudaEvent_t> end_events(bufs.size());
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventCreate(&start_events[rank]));
      CUDA_CHECK(cudaEventCreate(&end_events[rank]));
      CUDA_CHECK(cudaEventRecord(start_events[rank], bufs[rank].stream));
      CUDA_CHECK(cudaGraphLaunch(graph_execs[rank], bufs[rank].stream));
      CUDA_CHECK(cudaEventRecord(end_events[rank], bufs[rank].stream));
    }
    custom_allreduce.sync_streams(bufs);

    double avg_us = 0.0;
    for (int rank = 0; rank < static_cast<int>(bufs.size()); ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_events[rank], end_events[rank]));
      avg_us += static_cast<double>(elapsed_ms) * 1000.0 / static_cast<double>(kBenchmarkIters);
      CUDA_CHECK(cudaEventDestroy(start_events[rank]));
      CUDA_CHECK(cudaEventDestroy(end_events[rank]));
      CUDA_CHECK(cudaGraphExecDestroy(graph_execs[rank]));
      CUDA_CHECK(cudaGraphDestroy(graphs[rank]));
    }
    avg_us /= static_cast<double>(bufs.size());

    custom_allreduce.fill_inputs(bufs);
    custom_allreduce.launch(bufs);
    custom_allreduce.sync_streams(bufs);
    const bool pass = custom_allreduce.verify(bufs);
    overall_pass = overall_pass && pass;

    const size_t actual_bytes = static_cast<size_t>(bytes) * bufs.size();
    const double avg_sec = avg_us / 1.0e6;
    const double gb_per_s = avg_sec > 0.0 ? static_cast<double>(actual_bytes) / avg_sec / 1.0e9 : 0.0;
    rows.push_back(MetricRow{bytes, actual_bytes, gb_per_s, avg_us, pass});

    custom_allreduce.free_buffers(bufs);
  }

  std::printf("{\n");
  std::printf("  \"Correctness\": \"%s\",\n", overall_pass ? "PASS" : "FAIL");
  std::printf("  \"data_size_unit\": \"bytes\",\n");
  std::printf("  \"throughput_unit\": \"GB/s\",\n");
  std::printf("  \"latency_unit\": \"us\",\n");
  std::printf("  \"metrics\": [\n");
  for (size_t i = 0; i < rows.size(); ++i) {
    const MetricRow& row = rows[i];
    std::printf("    {\"data_size\": %d, \"actual_bytes\": %zu, \"throughput_avg\": %.3f, "
                "\"latency_avg\": %.3f, \"pass\": %s}",
                row.bytes, row.actual_bytes, row.throughput_avg_gb_per_s, row.latency_avg_us,
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
      if (value != "pull") {
        std::fprintf(stderr, "Unknown --mode '%s'; this example supports --mode pull\n", value.c_str());
        return EXIT_FAILURE;
      }
    } else if (arg == "--gpus" && i + 1 < argc) {
      requested_gpus = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "Usage: %s [--gpus N] [--mode pull]\n", argv[0]);
      return EXIT_FAILURE;
    }
  }

  int available_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
  const int num_gpus = std::min({requested_gpus, available_gpus, kMaxGpus});
  if (!(num_gpus == 1 || num_gpus == 2 || num_gpus == 4 || num_gpus == 8)) {
    std::fprintf(stderr, "This standalone all-reduce test supports 1, 2, 4, or 8 GPUs; got %d\n", num_gpus);
    return EXIT_FAILURE;
  }

  CustomAllReducePullStandalone custom_allreduce(num_gpus);
  benchmark(custom_allreduce);
  return EXIT_SUCCESS;
}
