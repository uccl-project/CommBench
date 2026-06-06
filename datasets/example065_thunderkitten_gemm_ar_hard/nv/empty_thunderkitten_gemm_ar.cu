// HARD task — multi-GPU gemm_ar from scratch (no ThunderKittens).
//
// Behavioral spec (mirrors the reference at `ref_thunderkitten_gemm_ar.cu`):
// ThunderKittens BF16 multi-GPU GEMM + AllReduce (NVLink/multicast).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// `int main()` forks NUM_DEVICES (8) child processes, each pinned to one GPU.
// Each child uses the ThunderKittens VMM/IPC helpers (CUDA driver API)
// together with a POSIX-socket-backed Broker to allocate shareable
// device memory, exchange handles, and bring up an NVLink multicast region
// for the output tensor `C`.
//
// Note vs. the upstream `gemm_ar_h100*.cu`:
//   * Upstream is Hopper-only (uses `warpgroup::mma_AB` / `wgmma`). Blackwell
//     ISA does not have `wgmma`; this file ports the kernel to the
//     Blackwell-native cluster-of-2 `tcgen05.mma2` path, mirroring the
//     compute structure of `ag_gemm_b200.cu`.
//   * The H100 LCSC kernel interleaves the all-reduce inside the matmul
//     pipeline ("comm SM" + per-tile barrier signaling). Doing that on top
//     of the cluster-of-2 tcgen05 schedule is a substantially different
//     kernel; this file uses a clearer two-kernel sequence instead —
//     matmul kernel writes local partial C into a multicast tensor, then a
//     dedicated multimem all-reduce kernel sums in place. Slightly less
//     perf than the H100 fully-fused version but byte-for-byte correct
//     and the numerical bandwidth matches upstream's `comm SM` formula.
//
// Code layout:
//   * Device-side kernels — `matmul::kernel` (cluster-of-2 tcgen05 GEMM
//     into the local shard of `C`), `all_reduce::kernel` (multimem.ld_reduce
//     + multimem.st on the multicast `C`), `barrier::kernel` (cross-device
//     rendezvous).
//   * `ParallelTensor` — pure-C++ replacement for TKParallelTensor.
//   * `DeviceBuffer` — RAII cudaMalloc wrapper for the non-shared A / B.
//   * `MatmulAllReduce` — clean class. Owns the Broker and
//     exposes only `run` / `sync`.
//   * `runTest(...)` — correctness check + TFLOP/s benchmark.
//   * `printJsonResult(...)` — emits exactly one JSON object on rank 0.
//   * `rank_main` / `main` — fork NUM_DEVICES children and wait.
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
// The reference at `ref_thunderkitten_gemm_ar.cu` is the behavioral /
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
//   Matmul kernel — cluster-of-2 Blackwell tcgen05 GEMM.
//
//   Mirrors the compute structure of `ag_gemm_b200.cu`:
//     * 2-CTA cluster, 148 blocks, 1 producer + 1 consumer warpgroup
//     * pipelined TMA loads of A (split row-wise across cluster) and B
//     * `mm2_ABt`/`mma2_ABt` into a shared tt<float, 256, 256> accumulator
//     * 8-stage epilogue that loads tensor memory, converts fp32→bf16,
//       and TMA-stores the result tile to local `C`.
//
//   Layout chosen to match the upstream `gemm_ar_h100.cu` interface:
//     * A: (M, K) bf16, plain global (per-rank slice of the K dim)
//     * B: (N, K) bf16, plain global (so the GEMM is `A @ B^T`, matching
//       gemm_rs_b200 / ag_gemm_b200's `mm2_ABt` orientation — the host
//       reshapes the upstream `(K, N)` weight into `(N, K)` before launch)
//     * C: (M, N) bf16, local cudaMalloc'd buffer (we add the multicast-
//       tensor wrapper as a separate `Cmc` so the all-reduce kernel can
//       reduce in-place across ranks).
// =====================================================================

namespace matmul {

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

__device__ inline void kernel(const globals &g) {
    // TODO
}

} // namespace matmul

// =====================================================================
//   All-reduce kernel — multimem.ld_reduce + multimem.st over multicast C.
//   (Same structure as example48 / `all_reduce::kernel`.)
// =====================================================================

namespace all_reduce {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;

    // TODO: define remaining kernel-launch shape parameters
    //       (NUM_THREADS, DYNAMIC_SHARED_MEMORY, etc.).
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 2;

    // TODO: define the kernel's globals (TMA descriptors,
    //       peer-pointer arrays, dev_idx, runtime params, plus
    //       any host-side helpers used by the launcher such as
    //       `dim3 grid()` / `int dynamic_shared_memory()`).
};

__device__ inline void kernel(const globals &G) {
    // TODO
}

} // namespace all_reduce

// =====================================================================
//   Cross-device barrier kernel.
// =====================================================================

namespace bracket_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;

    // TODO: define remaining kernel-launch shape parameters
    //       (NUM_THREADS, DYNAMIC_SHARED_MEMORY, etc.).
};

struct globals {
    static constexpr int NUM_DEVICES = 8;

    // TODO: define the kernel's globals (TMA descriptors,
    //       peer-pointer arrays, dev_idx, runtime params, plus
    //       any host-side helpers used by the launcher such as
    //       `dim3 grid()` / `int dynamic_shared_memory()`).
};

__device__ inline void kernel(const globals &G) {
    // TODO
}

} // namespace bracket_barrier


// =====================================================================
//   Launch helper. Supports CLUSTER_SIZE > 1 (uses cudaLaunchKernelEx +
//   kittens::LaunchConfig).
// =====================================================================

template <typename C>
consteval int kernel_min_blocks_per_sm() {
    if constexpr (requires { C::MIN_BLOCKS_PER_SM; })
        return static_cast<int>(C::MIN_BLOCKS_PER_SM);
    else
        return 1;
}

template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, kernel_min_blocks_per_sm<Config>())
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
//   ParallelTensor (same as example48).
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
//   DeviceBuffer (cudaMalloc RAII for A and B).
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
//   Init kernel for the correctness check.
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
//   Reshape helper — the matmul kernel computes A @ B^T (fed B as (N, K)),
//   so when the upstream contract has B as (K, N) we transpose at fill
//   time. For our correctness fill of B = const, the transpose is a
//   no-op; the wrapper makes this explicit.
// =====================================================================

// =====================================================================
//   MatmulAllReduce — clean communication class.
// =====================================================================

class MatmulAllReduce {
 public:
    MatmulAllReduce(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    MatmulAllReduce(const MatmulAllReduce &) = delete;
    MatmulAllReduce &operator=(const MatmulAllReduce &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    Broker &broker() { return broker_; }
    void sync() { broker_.sync(); }

    // Two bracketed kernel launches: matmul → all-reduce.
    //
    //   * `A`        : (M, K) bf16 plain device buffer (per-rank slice of K).
    //   * `B`        : (N, K) bf16 plain device buffer — note transposed
    //                  shape relative to the upstream `(K, N)` convention.
    //                  We compute `A @ B^T` so `(M, K) @ (K, N) = (M, N)`.
    //   * `C`        : (M, N) bf16 multicast tensor; matmul writes the local
    //                  partial, then all-reduce sums in place across ranks.
    //   * `barrier`  : 1 int multicast — cross-device rendezvous.
    void run(DeviceBuffer &A, DeviceBuffer &B, ParallelTensor &C,
             ParallelTensor &barrier, int M, int K, int N) {
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
//   Test harness (correctness + performance benchmark).
// =====================================================================

struct MetricRow {
    double data_size_mb;       // C output footprint in MB (M*N*2/1e6)
    double throughput_tflops;  // 2*M*N*K / time, in TFLOP/s
    double latency_ms;
};

// (M=size, K=size/world_size, N=size). 7 distinct sizes — exceeds the 6
// the example asks for. Sizes mirror the upstream `benchmark.py` shapes.
static constexpr int kBenchmarkSizes[] = {2048, 3072, 4096, 6144, 8192, 12288, 16384};
static constexpr int kNumWarmupIters  = 1;
static constexpr int kNumIters        = 5;
static constexpr int kCorrectnessSize = 2048;

// Correctness: A and B both filled with 1.0 on every rank → per-rank
// `C_r[i,j] = K = size/W`. After the all-reduce, each rank sees
// `C[i,j] = K * W = size`. We check three sample positions with a 2%
// tolerance (bf16 spacing at `size = 2048` is comfortably below this).
static std::pair<bool, std::vector<MetricRow>> runTest(
    MatmulAllReduce &comm,
    const int *sizes,
    int num_sizes,
    int correctness_size,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    const int W = comm.world_size();

    auto fill = [&](void *p, size_t numel, __nv_bfloat16 v) {
        const int threads = 256;
        const dim3 grid((numel + threads - 1) / threads);
        fill_bf16_kernel<<<grid, threads>>>(reinterpret_cast<__nv_bfloat16 *>(p), numel, v);
    };

    auto run_once = [&](int M, int K, int N,
                        DeviceBuffer &A, DeviceBuffer &B, ParallelTensor &C,
                        ParallelTensor &barrier) {
        comm.run(A, B, C, barrier, M, K, N);
    };

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int M = correctness_size;
        const int K = correctness_size / W;
        const int N = correctness_size;
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_bfloat16);  // (N,K)
        const size_t C_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = sizeof(int);

        DeviceBuffer A(A_bytes);
        DeviceBuffer B(B_bytes);
        ParallelTensor C(comm.broker(), C_bytes, comm.rank(), W, true);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Seeded, non-constant, non-zero inputs (every rank fills identically).
        // A is row-constant (A[i,k] = a_vec[k], varying with k) so the
        // all-reduced output C[i,j] = W * sum_k a_vec[k]*B[j,k] depends only on
        // the column j. The CPU reference is therefore independent of row.
        std::vector<float> a_vec(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k) a_vec[k] = tk_pattern(static_cast<unsigned long long>(k), 0x1111ULL);
        std::vector<__nv_bfloat16> hA(static_cast<size_t>(M) * K);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                hA[static_cast<size_t>(i) * K + k] = __float2bfloat16(a_vec[k]);
        std::vector<float> b_host(static_cast<size_t>(N) * K);
        std::vector<__nv_bfloat16> hB(static_cast<size_t>(N) * K);
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k) {
                float bv = tk_pattern(static_cast<unsigned long long>(j) * K + k, 0x2222ULL);
                b_host[static_cast<size_t>(j) * K + k] = bv;
                hB[static_cast<size_t>(j) * K + k] = __float2bfloat16(bv);
            }
        MatmulAllReduce::cuda_check(cudaMemcpy(A.ptr(), hA.data(), A_bytes, cudaMemcpyHostToDevice));
        MatmulAllReduce::cuda_check(cudaMemcpy(B.ptr(), hB.data(), B_bytes, cudaMemcpyHostToDevice));
        MatmulAllReduce::cuda_check(cudaMemsetAsync(C.raw_ptrs[comm.rank()], 0, C_bytes));
        MatmulAllReduce::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, barrier_bytes));
        MatmulAllReduce::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        run_once(M, K, N, A, B, C, barrier);
        MatmulAllReduce::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        const size_t sample_idxs[3] = {
            0,
            static_cast<size_t>(M / 2) * N + (N / 2),
            static_cast<size_t>(M - 1) * N + (N - 1),
        };
        // Sharding-agnostic reference: C[*,col] = W * sum_k a_vec[k]*B[col,k].
        auto expected_at = [&](size_t local_idx) -> float {
            size_t col = local_idx % static_cast<size_t>(N);
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<double>(a_vec[k]) * static_cast<double>(b_host[col * K + k]);
            return static_cast<float>(static_cast<double>(W) * acc);
        };
        __nv_bfloat16 host[3] = {};
        for (int i = 0; i < 3; ++i) {
            MatmulAllReduce::cuda_check(cudaMemcpy(
                &host[i],
                reinterpret_cast<__nv_bfloat16 *>(C.raw_ptrs[comm.rank()]) + sample_idxs[i],
                sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));
        }
        overall_pass = true;
        for (int i = 0; i < 3; ++i) {
            float v = __bfloat162float(host[i]);
            float exp = expected_at(sample_idxs[i]);
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
        const int K = size / W;
        const int N = size;
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_bfloat16);
        const size_t C_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = sizeof(int);

        DeviceBuffer A(A_bytes);
        DeviceBuffer B(B_bytes);
        ParallelTensor C(comm.broker(), C_bytes, comm.rank(), W, true);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Non-zero, non-constant inputs so a kernel cannot fake fast timing by
        // detecting all-zero data and skipping the matmul.
        fill_pattern_kernel<<<dim3((A_bytes / sizeof(__nv_bfloat16) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(A.ptr()), A_bytes / sizeof(__nv_bfloat16), 0x1111ULL);
        fill_pattern_kernel<<<dim3((B_bytes / sizeof(__nv_bfloat16) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(B.ptr()), B_bytes / sizeof(__nv_bfloat16), 0x2222ULL);
        MatmulAllReduce::cuda_check(cudaMemsetAsync(C.raw_ptrs[comm.rank()], 0, C_bytes));
        MatmulAllReduce::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, barrier_bytes));
        MatmulAllReduce::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            run_once(M, K, N, A, B, C, barrier);
        MatmulAllReduce::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        MatmulAllReduce::cuda_check(cudaEventCreate(&start_evt));
        MatmulAllReduce::cuda_check(cudaEventCreate(&stop_evt));
        MatmulAllReduce::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            run_once(M, K, N, A, B, C, barrier);
        MatmulAllReduce::cuda_check(cudaEventRecord(stop_evt));
        MatmulAllReduce::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        MatmulAllReduce::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        MatmulAllReduce::cuda_check(cudaEventDestroy(start_evt));
        MatmulAllReduce::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms = total_ms / iters;
        const double total_flops = 2.0 * static_cast<double>(M) * N * K;
        const double tflops = avg_ms > 0.0
            ? total_flops * 1e-12 / (avg_ms * 1e-3)
            : 0.0;

        rows.push_back({
            static_cast<double>(C_bytes) / 1e6,
            tflops,
            avg_ms,
        });

        comm.sync();
    }

    return {overall_pass, rows};
}


// =====================================================================
//   JSON output.
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
//   Per-rank child entry point.
// =====================================================================

static int rank_main(int rank, int world_size) {
    if (cudaSetDevice(rank) != cudaSuccess) {
        std::fprintf(stderr, "rank %d: cudaSetDevice failed\n", rank);
        return 1;
    }
    cudaFree(0);

    MatmulAllReduce comm(rank, world_size);

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
//   main — fork one process per GPU and wait.
// =====================================================================

int main(int /*argc*/, char ** /*argv*/) {
    constexpr int WORLD_SIZE = all_reduce::globals::NUM_DEVICES;

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
