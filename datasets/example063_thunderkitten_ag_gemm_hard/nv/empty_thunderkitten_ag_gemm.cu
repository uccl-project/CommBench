// HARD task — multi-GPU ag_gemm from scratch (no ThunderKittens).
//
// Behavioral spec (mirrors the reference at `ref_thunderkitten_ag_gemm.cu`):
// ThunderKittens BF16 multi-GPU All-Gather + GEMM (NVLink/multicast).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// `int main()` forks NUM_DEVICES (8) child processes, each pinned to one GPU.
// Each child uses the ThunderKittens VMM/IPC helpers (CUDA driver API)
// together with a POSIX-socket-backed Broker to allocate shareable
// device memory, exchange handles, and bring up an NVLink multicast region
// for the gathered activation tensor `A`.
//
// The kernel is the upstream `ag_gemm_b200.cu` (Blackwell tcgen05 path)
// — it works unchanged on B300 (compute_103a). H100 (compute_90a) is not
// targeted by this file; use the upstream H100 kernel for that.
//
// Code layout (per the dataset readme guidelines):
//   * Device-side kernels — `comm_sm` (TMA gathers per-rank shards into the
//     multicast `A`, signals per-row barriers), `comp_sm` (cluster-of-2
//     producer/consumer GEMM with tcgen05.mma2 + 8-way epilogue), wrapped
//     by `main_kernel` and `epilogue_kernel`.
//   * `ParallelTensor` — pure-C++ replacement for TKParallelTensor;
//     allocates VMM-backed shareable physical memory, exchanges POSIX fds,
//     and (optionally) lays a multicast handle on top.
//   * `DeviceBuffer` — thin RAII wrapper around cudaMalloc for the
//     non-shared B / C tensors.
//   * `AllGatherMatmul` — clean communication class. Owns the Broker
//     and exposes only the core ops (run / sync). No test or benchmark
//     logic embedded.
//   * `runTest(...)` — dedicated standalone function. Runs a deterministic
//     correctness check at a small size and a TFLOP/s benchmark across
//     `kBenchmarkSizes` (1 warmup + kNumIters timed iterations per size).
//   * `printJsonResult(...)` — emits exactly one JSON object on rank 0.
//   * `rank_main` / `main` — fork NUM_DEVICES children, wait, and propagate
//     a non-zero exit if any rank failed correctness or crashed.
//
// =================== HARD-TASK CONTRACT ===================
// The full test harness, the comm-class API, and the *usage* of every
// interface have been preserved below — you can see exactly how the
// classes / kernels are called from `runTest`. Your job is to implement
// the TODO-marked bodies (Broker, ParallelTensor, the device kernels,
// the launch helper, and the kernel-launching methods of the comm class)
// using only base libraries.
//
// You MUST implement this WITHOUT depending on ThunderKittens.
// Only the following may be `#include`d:
//   * CUDA driver API   — `<cuda.h>`
//   * CUDA runtime API  — `<cuda_runtime.h>`
//   * BF16 / FP8 intrinsics — `<cuda_bf16.h>`, `<cuda_fp8.h>` (as needed)
//   * POSIX             — `<fcntl.h>`, `<signal.h>`, `<sys/mman.h>`,
//                          `<sys/prctl.h>`, `<sys/wait.h>`, `<unistd.h>`,
//                          `<sys/socket.h>`, `<sys/un.h>`, `<poll.h>`, ...
//   * The C++ standard library
//
// FORBIDDEN: any header from `kittens.cuh`, `pyutils/`, `types/system/`,
// `prototype.cuh` — any path under `third_party/ThunderKittens/`. Do NOT
// link NCCL, NVSHMEM, MPI, PyTorch, or pybind either.
//
// Implementation hints:
//   * VMM-backed shareable memory: `cuMemCreate` + `cuMemMap` +
//     `cuMemSetAccess` with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`.
//   * Cross-rank handle exchange: a POSIX SHM mailbox +
//     Unix-domain-socket SCM_RIGHTS FD passing.
//   * Multicast: `cuMulticastCreate` + `cuMulticastBindMem` +
//     `cuMulticastBindDevice` (NVLink Switch only).
//   * Device kernels: inline PTX (`cp.async.bulk.tensor.*` for TMA,
//     `multimem.*` for NVLS, `tcgen05.mma` for tensor-memory MMA on
//     Blackwell) or roll your own equivalents.
//   * Cross-device barrier: NVLS atomic counter or per-rank flag spinwait.
//
// The reference at `ref_thunderkitten_ag_gemm.cu` is the behavioral /
// numerical spec — your generated file must produce the same JSON output
// schema and comparable throughput / latency, but it MUST NOT include any
// ThunderKittens header.


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>



// =====================================================================
//   ThunderKittens-equivalent constants used by the preserved kernel
//   `config` / `globals` declarations. The reference relies on these
//   coming from `kittens.cuh`; we inline them so the file still compiles
//   once the AI fills in the TODO bodies. (Values match the upstream
//   kittens header for sm_100 / sm_103.)
// =====================================================================
namespace {
inline constexpr int WARP_THREADS      = 32;
inline constexpr int WARPGROUP_WARPS   = 4;
inline constexpr int WARPGROUP_THREADS = WARP_THREADS * WARPGROUP_WARPS;
inline constexpr int MAX_SHARED_MEMORY = 227 * 1024;  // sm_100/103 max smem per SM
}


// =====================================================================
//   Broker — TODO: cross-rank coordination for POSIX FD exchange.
//
//   The reference uses ThunderKittens' KittensBroker. Implement a
//   from-scratch equivalent (POSIX-SHM mailbox + Unix-domain-socket
//   SCM_RIGHTS FD passing) with at minimum:
//
//     class Broker {
//      public:
//         Broker(int rank, int world_size);
//         void sync();                                       // barrier
//         void exchange_fds(int *all_fds, int my_fd);        // all-gather of FDs
//         void broadcast_fd(int *out_fd, int src_fd, int src_rank);
//     };
// =====================================================================

class Broker {
 public:
    Broker(int /*rank*/, int /*world_size*/) {
        // TODO
    }
    Broker(const Broker &) = delete;
    Broker &operator=(const Broker &) = delete;

    void sync() {
        // TODO
    }

    void exchange_fds(int * /*all_fds*/, int /*my_fd*/) {
        // TODO
    }

    void broadcast_fd(int * /*out_fd*/, int /*src_fd*/, int /*src_rank*/) {
        // TODO
    }
};

// =====================================================================
//   Device-side kernels (kernel bodies are byte-for-byte the upstream
//   ag_gemm_b200.cu — perf must be unchanged)
// =====================================================================

struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    // TODO: define remaining kernel-launch shape parameters
    //       (NUM_THREADS, DYNAMIC_SHARED_MEMORY, etc.).
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int PIPELINE_STAGES = 5;
    static constexpr int MMA_PIPE_DEPTH = 2;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    // TODO: define the kernel's globals (TMA descriptors,
    //       peer-pointer arrays, dev_idx, runtime params, plus
    //       any host-side helpers used by the launcher such as
    //       `dim3 grid()` / `int dynamic_shared_memory()`).
};

__device__ inline void comm_sm(const globals &g) {
    // TODO
}

__device__ inline void comp_sm(const globals &g) {
    // TODO
}

__device__ inline void main_kernel(const globals &g) {
    // TODO
}

__device__ inline void epilogue_kernel(const globals &g) {
    // TODO
}


// =====================================================================
//   Minimal launch helper. Supports CLUSTER_SIZE >= 1 (uses
//   cudaLaunchKernelEx with kittens::LaunchConfig for clusters).
// =====================================================================

template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS)
void global_kernel(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Config, typename Globals, auto Kernel>
__host__ inline void launch_kernel(const Globals &G, cudaStream_t stream = 0) {
    // TODO: launch the kernel (cudaFuncSetAttribute for dynamic
    //       shared memory, then `global_kernel<<<grid, block, smem,
    //       stream>>>(G)` with cudaGetLastError check).
}


// =====================================================================
//   ParallelTensor — pure-C++ replacement for TKParallelTensor
// =====================================================================

class ParallelTensor {
 public:
    static constexpr int MAX_DEVICES = 8;

    void *raw_ptrs[MAX_DEVICES] = {};
    void *mc_ptr = nullptr;
    size_t allocated_size = 0;
    size_t mc_allocated_size = 0;
    int local_rank = -1;
    int local_world_size = -1;
    bool multicast = false;

    ParallelTensor(Broker &broker, size_t bytes, int rank, int world_size, bool mc)
        : local_rank(rank), local_world_size(world_size), multicast(mc) {
        // TODO: implement (uses cuMemCreate / cuMemMap /
        //       cuMemSetAccess + POSIX-FD-passing IPC + optional
        //       cuMulticast* in the reference).
    }

    ParallelTensor(const ParallelTensor &) = delete;
    ParallelTensor &operator=(const ParallelTensor &) = delete;
    ~ParallelTensor() = default;

    void initialize_multicast(Broker &broker) {
        // TODO: implement (uses cuMemCreate / cuMemMap /
        //       cuMemSetAccess + POSIX-FD-passing IPC + optional
        //       cuMulticast* in the reference).
    }
};


// =====================================================================
//   DeviceBuffer — RAII cudaMalloc wrapper for non-shared B / C tensors.
// =====================================================================

class DeviceBuffer {
 public:
    DeviceBuffer() = default;
    DeviceBuffer(size_t bytes) { alloc(bytes); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }

    void alloc(size_t bytes) {
        if (cudaMalloc(&ptr_, bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        size_ = bytes;
    }

    void *ptr() const { return ptr_; }
    size_t size() const { return size_; }

 private:
    void *ptr_ = nullptr;
    size_t size_ = 0;
};


// =====================================================================
//   Init kernel for the correctness check
// =====================================================================

__global__ void fill_bf16_kernel(__nv_bfloat16 *p, size_t n, __nv_bfloat16 v) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// Deterministic, non-constant, non-zero test pattern. Values are drawn from
// {0.5, 1.0, 1.5, 2.0} — all exactly representable in bf16 — so a correctness
// reference can be computed in closed form while inputs remain unpredictable.
// This defeats kernels that try to pass by detecting a trivial input (all-zero
// or all-equal) or by hardcoding a constant expected output.
__host__ __device__ inline float tk_pattern(unsigned long long idx, unsigned long long seed) {
    unsigned long long x = (idx + 1ULL) * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x ^= x >> 31;
    return 0.5f * static_cast<float>(1u + static_cast<unsigned>(x & 3ULL));
}

__global__ void fill_pattern_kernel(__nv_bfloat16 *p, size_t n, unsigned long long seed) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2bfloat16(tk_pattern(i, seed));
}


// =====================================================================
//   AllGatherMatmul — clean communication class (no test/benchmark logic).
//
//   Owns the Broker and exposes only the core ops:
//     run, sync.
// =====================================================================

class AllGatherMatmul {
 public:
    AllGatherMatmul(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    AllGatherMatmul(const AllGatherMatmul &) = delete;
    AllGatherMatmul &operator=(const AllGatherMatmul &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    Broker &broker() { return broker_; }

    void sync() { broker_.sync(); }

    // One bracketed all-gather + matmul launch.
    //
    //   * `A`        : (M, K) bf16, multicast=true (each rank owns the row
    //                  shard [rank*M/W, (rank+1)*M/W); kernel gathers the
    //                  rest in-place via TMA over the multicast handle).
    //   * `B`        : (N, K) bf16, plain device buffer, fully replicated
    //                  on every rank (kernel computes A @ B^T).
    //   * `C`        : (M, N) bf16, plain device buffer, written locally.
    //   * `barrier`  : (2, 1024, 1024) int multicast, cleared by epilogue.
    void run(ParallelTensor &A, DeviceBuffer &B, DeviceBuffer &C,
             ParallelTensor &barrier, int M, int K, int N, int num_comm_sms) {
        // TODO: implement using your kernel + Broker + ParallelTensor.
        //       (See `ref_thunderkitten_*.cu` for the upstream version.)
    }

    static void cuda_check(cudaError_t err) {
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err));
    }

 private:
    int rank_;
    int world_size_;
    Broker broker_;
};


// =====================================================================
//   Test harness (correctness + performance benchmark)
// =====================================================================

struct MetricRow {
    double data_size_mb;       // input A footprint in MB (M*K*2/1e6)
    double throughput_tflops;  // 2*M*N*K / time, in TFLOP/s
    double latency_ms;
};

// Each entry is (M=K=size, N=size/world_size). Matches the upstream
// `benchmark.py` size sweep — `num_comm_sms` is fixed at `kNumCommSms`
// instead of the upstream 6-value sweep, so the example finishes in
// roughly a minute on B300.
static constexpr int kBenchmarkSizes[] = {2048, 4096, 8192, 16384, 32768};
static constexpr int kNumCommSms      = 16;
static constexpr int kNumWarmupIters  = 1;
static constexpr int kNumIters        = 5;
static constexpr int kCorrectnessSize = 2048;  // smallest that satisfies kernel block constraints


// Standalone test/benchmark function — handles correctness + timing.
//
//   * Correctness: each rank fills its A_local shard with (rank+1)/K and B
//     with 1.0. After the all-gather + GEMM, on every rank
//        C[i, j] = sum_k A_full[i, k] * B[j, k] = i_block + 1
//     where i_block = i / (M/W). We sample C[r * (M/W), 0] for every
//     r ∈ [0, W) and check it equals (r+1) within 1e-1.
//   * Benchmark: for each (M=K=size, N=size/W) in `sizes`, run
//     `warmup_iters` warmup iterations + `iters` cudaEvent-timed iterations.
//     Throughput uses 2*M*N*K / time in TFLOP/s.
static std::pair<bool, std::vector<MetricRow>> runTest(
    AllGatherMatmul &comm,
    const int *sizes,
    int num_sizes,
    int correctness_size,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    const int W = comm.world_size();

    auto run_once = [&](int M, int K, int N,
                        ParallelTensor &A, DeviceBuffer &B, DeviceBuffer &C,
                        ParallelTensor &barrier) {
        comm.run(A, B, C, barrier, M, K, N, kNumCommSms);
    };

    auto fill_shard = [&](void *p, size_t numel, __nv_bfloat16 v) {
        const int threads = 256;
        const dim3 grid((numel + threads - 1) / threads);
        fill_bf16_kernel<<<grid, threads>>>(reinterpret_cast<__nv_bfloat16 *>(p), numel, v);
    };

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int M = correctness_size;
        const int K = correctness_size;
        const int N = correctness_size / W;
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_bfloat16);
        const size_t C_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = 2u * 1024u * 1024u * sizeof(int);

        ParallelTensor A(comm.broker(), A_bytes, comm.rank(), W, true);
        DeviceBuffer B(B_bytes);
        DeviceBuffer C(C_bytes);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Seeded, non-constant, non-zero inputs. Each rank fills its A row shard
        // [r*M/W, (r+1)*M/W) with a GLOBAL-row-indexed pattern, and B (replicated)
        // with another pattern. After all-gather + GEMM,
        //   C[i,j] = sum_k A_full[i,k] * B[j,k]
        // is fully data-dependent, so a kernel cannot pass by hardcoding or by
        // detecting trivial (all-zero / all-equal) inputs.
        const int shard_rows = M / W;
        const size_t shard_offset = static_cast<size_t>(comm.rank()) * shard_rows * K;
        std::vector<__nv_bfloat16> hA(static_cast<size_t>(shard_rows) * K);
        for (int lr = 0; lr < shard_rows; ++lr) {
            int grow = comm.rank() * shard_rows + lr;
            for (int k = 0; k < K; ++k)
                hA[static_cast<size_t>(lr) * K + k] =
                    __float2bfloat16(tk_pattern(static_cast<unsigned long long>(grow) * K + k, 0x1111ULL));
        }
        AllGatherMatmul::cuda_check(cudaMemcpy(
            reinterpret_cast<__nv_bfloat16 *>(A.raw_ptrs[comm.rank()]) + shard_offset,
            hA.data(), static_cast<size_t>(shard_rows) * K * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        std::vector<float> b_host(static_cast<size_t>(N) * K);
        std::vector<__nv_bfloat16> hB(static_cast<size_t>(N) * K);
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k) {
                float bv = tk_pattern(static_cast<unsigned long long>(j) * K + k, 0x2222ULL);
                b_host[static_cast<size_t>(j) * K + k] = bv;
                hB[static_cast<size_t>(j) * K + k] = __float2bfloat16(bv);
            }
        AllGatherMatmul::cuda_check(cudaMemcpy(B.ptr(), hB.data(), B_bytes, cudaMemcpyHostToDevice));
        AllGatherMatmul::cuda_check(cudaMemsetAsync(C.ptr(), 0, C_bytes));
        AllGatherMatmul::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, barrier_bytes));
        AllGatherMatmul::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        run_once(M, K, N, A, B, C, barrier);
        AllGatherMatmul::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        // Reference: C[row,col] = sum_k tk_pattern_A(row,k) * B[col,k].
        // Sample C[r*(M/W), 0] for every r (known global rows, column 0).
        auto expected_at = [&](size_t row, size_t col) -> float {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<double>(tk_pattern(row * K + k, 0x1111ULL)) *
                       static_cast<double>(b_host[col * K + k]);
            return static_cast<float>(acc);
        };
        std::vector<__nv_bfloat16> samples(W);
        for (int r = 0; r < W; ++r) {
            const size_t idx = static_cast<size_t>(r) * (M / W) * N;  // column 0 of row r*(M/W)
            AllGatherMatmul::cuda_check(cudaMemcpy(
                &samples[r],
                reinterpret_cast<__nv_bfloat16 *>(C.ptr()) + idx,
                sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));
        }
        overall_pass = true;
        for (int r = 0; r < W; ++r) {
            float v = __bfloat162float(samples[r]);
            float exp = expected_at(static_cast<size_t>(r) * (M / W), 0);
            if (std::fabs(v - exp) > std::fabs(exp) * 0.05f + 1e-3f) {
                overall_pass = false;
                break;
            }
        }
        comm.sync();
    }

    // ---- benchmark ----
    for (int i = 0; i < num_sizes; ++i) {
        const int size = sizes[i];
        const int M = size;
        const int K = size;
        const int N = size / W;
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_bfloat16);
        const size_t C_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = 2u * 1024u * 1024u * sizeof(int);

        ParallelTensor A(comm.broker(), A_bytes, comm.rank(), W, true);
        DeviceBuffer B(B_bytes);
        DeviceBuffer C(C_bytes);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Non-zero, non-constant inputs so a kernel cannot fake fast timing by
        // detecting all-zero data and skipping the all-gather + matmul.
        fill_pattern_kernel<<<dim3((A_bytes / sizeof(__nv_bfloat16) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(A.raw_ptrs[comm.rank()]), A_bytes / sizeof(__nv_bfloat16), 0x1111ULL);
        fill_pattern_kernel<<<dim3((B_bytes / sizeof(__nv_bfloat16) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(B.ptr()), B_bytes / sizeof(__nv_bfloat16), 0x2222ULL);
        AllGatherMatmul::cuda_check(cudaMemsetAsync(C.ptr(), 0, C_bytes));
        AllGatherMatmul::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, barrier_bytes));
        AllGatherMatmul::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            run_once(M, K, N, A, B, C, barrier);
        AllGatherMatmul::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        AllGatherMatmul::cuda_check(cudaEventCreate(&start_evt));
        AllGatherMatmul::cuda_check(cudaEventCreate(&stop_evt));
        AllGatherMatmul::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            run_once(M, K, N, A, B, C, barrier);
        AllGatherMatmul::cuda_check(cudaEventRecord(stop_evt));
        AllGatherMatmul::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        AllGatherMatmul::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        AllGatherMatmul::cuda_check(cudaEventDestroy(start_evt));
        AllGatherMatmul::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms = total_ms / iters;
        const double total_flops = 2.0 * static_cast<double>(M) * N * K;
        const double tflops = avg_ms > 0.0
            ? total_flops * 1e-12 / (avg_ms * 1e-3)
            : 0.0;

        rows.push_back({
            static_cast<double>(A_bytes) / 1e6,
            tflops,
            avg_ms,
        });

        comm.sync();
    }

    return {overall_pass, rows};
}


// =====================================================================
//   JSON output (rank 0 only, must be exactly one JSON object)
// =====================================================================

static void printJsonResult(bool overall_pass, const std::vector<MetricRow> &rows) {
    std::cout << "{\n";
    std::cout << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL") << "\",\n";
    std::cout << "  \"data_size_unit\": \"MB\",\n";
    std::cout << "  \"throughput_unit\": \"TFLOPs\",\n";
    std::cout << "  \"latency_unit\": \"ms\",\n";
    std::cout << "  \"metrics\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto &r = rows[i];
        std::cout << "    {\"data_size\": " << r.data_size_mb
                  << ", \"throughput_avg\": " << r.throughput_tflops
                  << ", \"latency_avg\": " << r.latency_ms << "}";
        if (i + 1 != rows.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}


// =====================================================================
//   Per-rank child entry point
// =====================================================================

static int rank_main(int rank, int world_size) {
    if (cudaSetDevice(rank) != cudaSuccess) {
        std::fprintf(stderr, "rank %d: cudaSetDevice failed\n", rank);
        return 1;
    }
    cudaFree(0);

    AllGatherMatmul comm(rank, world_size);

    auto [correctness, rows] = runTest(
        comm,
        kBenchmarkSizes,
        static_cast<int>(sizeof(kBenchmarkSizes) / sizeof(kBenchmarkSizes[0])),
        kCorrectnessSize,
        kNumWarmupIters,
        kNumIters);

    comm.sync();

    if (rank == 0)
        printJsonResult(correctness, rows);

    return correctness ? 0 : 2;
}


// =====================================================================
//   main — fork one process per GPU and wait
// =====================================================================

int main(int /*argc*/, char ** /*argv*/) {
    constexpr int WORLD_SIZE = globals::NUM_DEVICES;

    shm_unlink("/kittens_broker_shm");

    pid_t pids[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            if (getppid() == 1) _exit(1);

            int rc = 0;
            try {
                rc = rank_main(i, WORLD_SIZE);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "rank %d: %s\n", i, e.what());
                rc = 1;
            }
            std::cout.flush();
            _exit(rc);
        }
        pids[i] = pid;
    }

    int overall_rc = 0;
    bool any_correctness_fail = false;
    for (int i = 0; i < WORLD_SIZE; ++i) {
        int status = 0;
        if (waitpid(pids[i], &status, 0) < 0) {
            std::perror("waitpid");
            overall_rc = 1;
            continue;
        }
        if (!WIFEXITED(status)) {
            std::fprintf(stderr, "rank %d: terminated abnormally\n", i);
            overall_rc = 1;
        } else {
            int rc = WEXITSTATUS(status);
            if (rc == 2) any_correctness_fail = true;
            else if (rc != 0) overall_rc = 1;
        }
    }

    if (any_correctness_fail) overall_rc = 1;
    return overall_rc;
}
