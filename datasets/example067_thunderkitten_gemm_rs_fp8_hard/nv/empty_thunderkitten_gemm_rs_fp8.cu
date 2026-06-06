// HARD task — multi-GPU gemm_rs_fp8 from scratch (no ThunderKittens).
//
// Behavioral spec (mirrors the reference at `ref_thunderkitten_gemm_rs_fp8.cu`):
// ThunderKittens FP8 multi-GPU GEMM + ReduceScatter (NVLink/multicast).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// `int main()` forks NUM_DEVICES (8) child processes, each pinned to one GPU.
// Each child uses the ThunderKittens VMM/IPC helpers (CUDA driver API)
// together with a POSIX-socket-backed Broker to allocate shareable
// device memory, exchange handles, and bring up the per-rank C shards
// over IPC (no multicast — the kernel TMA-stores tiles directly to the
// destination rank's shard with in-network ADD).
//
// The kernel is the upstream `gemm_rs_fp8_b200.cu` (Blackwell tcgen05.mma2
// in FP8 e4m3 + `warp::tma::store_add_async` of the bf16-converted output
// for the cross-device reduce-scatter); it works unchanged on B300
// (compute_103a). The upstream does not ship an H100 FP8 variant; this
// file does not target H100 either.
//
// Inputs `A`, `B` are e4m3 FP8; the output `C` is bf16. Note vs. the
// bf16 variant: `Kb = 128` (twice the bf16 reduction tile width).
//
// Code layout:
//   * Device-side kernels — `matmul_reduce_scatter::kernel` (cluster-of-2
//     tcgen05 GEMM + per-tile cross-device store-add to the destination
//     rank's C shard) and `matmul_reduce_scatter_barrier::kernel`
//     (multimem.red bracket barrier for end-of-iteration sync).
//   * `ParallelTensor` — pure-C++ replacement for TKParallelTensor.
//   * `DeviceBuffer` — RAII cudaMalloc wrapper for the non-shared A / B.
//   * `MatmulReduceScatter` — clean class. Owns the Broker and
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
// The reference at `ref_thunderkitten_gemm_rs_fp8.cu` is the behavioral /
// numerical spec — your generated file must produce the same JSON output
// schema and comparable throughput / latency, but it MUST NOT include any
// ThunderKittens header.


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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
//   gemm_rs_fp8_b200.cu — perf must be unchanged)
// =====================================================================

namespace matmul_reduce_scatter {

constexpr int NUM_CONSUMERS = (2);
constexpr int NUM_PRODUCERS = (1);

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 128;   // FP8: doubled reduction tile (vs. 64 for bf16)

static constexpr int NUM_DEVICES = 8;

struct globals {
    // TODO: define the kernel's globals (TMA descriptors,
    //       peer-pointer arrays, dev_idx, runtime params, plus
    //       any host-side helpers used by the launcher such as
    //       `dim3 grid()` / `int dynamic_shared_memory()`).
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int CLUSTER_M = 4*Mb, CLUSTER_N = Nb;

struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;

    // TODO: define remaining kernel-launch shape parameters
    //       (NUM_THREADS, DYNAMIC_SHARED_MEMORY, etc.).
};

__device__ static inline int get_iters_per_task(const globals &g) {
    // TODO
}
template<int SUPER_M=8> __device__ static inline int2 get_task_idx(const globals &g, int task_iter, bool is_consumer) {
    // TODO
}

__device__ void kernel(const globals &g) {
    // TODO
}

} // namespace matmul_reduce_scatter

namespace matmul_reduce_scatter_barrier {

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

} // namespace matmul_reduce_scatter_barrier


// =====================================================================
//   Launch helper. Supports CLUSTER_SIZE > 1.
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
//   ParallelTensor — pure-C++ replacement for TKParallelTensor.
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
// {0.5, 1.0, 1.5, 2.0} — all exactly representable in fp8 e4m3 — so a
// correctness reference can be computed in closed form while inputs stay
// unpredictable. Defeats kernels that pass by detecting a trivial input
// (all-zero / all-equal) or by hardcoding a constant expected output.
__host__ __device__ inline float tk_pattern(unsigned long long idx, unsigned long long seed) {
    unsigned long long x = (idx + 1ULL) * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x ^= x >> 31;
    return 0.5f * static_cast<float>(1u + static_cast<unsigned>(x & 3ULL));
}

__global__ void fill_pattern_fp8_kernel(__nv_fp8_e4m3 *p, size_t n, unsigned long long seed) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __nv_fp8_e4m3(tk_pattern(i, seed));
}


// =====================================================================
//   MatmulReduceScatter — clean communication class.
// =====================================================================

class MatmulReduceScatter {
 public:
    MatmulReduceScatter(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    MatmulReduceScatter(const MatmulReduceScatter &) = delete;
    MatmulReduceScatter &operator=(const MatmulReduceScatter &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    Broker &broker() { return broker_; }
    void sync() { broker_.sync(); }

    // One bracketed matmul + reduce-scatter launch.
    //
    //   * `A`        : (M, K) fp8e4m3 plain device buffer (per-rank slice of K).
    //   * `B`        : (N, K) fp8e4m3 plain device buffer (note transposed
    //                  shape vs. the upstream `(K, N)` convention; the
    //                  kernel uses `mm2_ABt`).
    //   * `C`        : (M / world_size, N) bf16 IPC tensor (per-rank shard
    //                  of the reduce-scattered output, in fp32→bf16
    //                  converted form). The kernel TMA-stores tiles to
    //                  the destination rank's shard with in-network ADD
    //                  (`store_add_async`), achieving reduce-scatter in a
    //                  single pass.
    //   * `barrier`  : 1 int multicast — final cross-device rendezvous.
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
    double data_size_mb;       // C output footprint per rank in MB ((M/W)*N*2/1e6)
    double throughput_tflops;  // 2*M*N*K / time, in TFLOP/s
    double latency_ms;
};

// (M=size, K=size/world_size, N=size). 7 distinct sizes; mirror the
// upstream `benchmark.py` shapes plus two intermediate points (3072,
// 6144, 12288) so the table has 7 rows — exceeds the 6-size requirement.
static constexpr int kBenchmarkSizes[] = {2048, 3072, 4096, 6144, 8192, 12288, 16384};
static constexpr int kNumWarmupIters  = 1;
static constexpr int kNumIters        = 5;
static constexpr int kCorrectnessSize = 2048;

// Correctness: A and B both filled with FP8 1.0 (byte 0x38) on every
// rank → per-rank A @ B^T yields `C_partial[i, j] = K`. After
// reduce-scatter, each rank receives the (M/W, N) shard summed across W
// ranks → expected value `K * W = size`. Sample three points with a 2%
// tolerance (bf16 spacing is comfortably below this for the test size).
static std::pair<bool, std::vector<MetricRow>> runTest(
    MatmulReduceScatter &comm,
    const int *sizes,
    int num_sizes,
    int correctness_size,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    const int W = comm.world_size();

    auto run_once = [&](int M, int K, int N,
                        DeviceBuffer &A, DeviceBuffer &B, ParallelTensor &C,
                        ParallelTensor &barrier) {
        comm.run(A, B, C, barrier, M, K, N);
    };

    // FP8 e4m3 byte for 1.0 — biased exponent 7 with mantissa 0 → 0x38.
    constexpr uint8_t kFp8_one_byte = 0x38;

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int M = correctness_size;
        const int K = correctness_size / W;
        const int N = correctness_size;
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_fp8_e4m3);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_fp8_e4m3);
        const size_t C_bytes = static_cast<size_t>(M / W) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = sizeof(int);

        DeviceBuffer A(A_bytes);
        DeviceBuffer B(B_bytes);
        ParallelTensor C(comm.broker(), C_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Seeded, non-constant, non-zero fp8 inputs (every rank fills identically).
        // A is row-constant (A[i,k] = a_vec[k], varying with k) so the
        // reduce-scattered output C[i,j] = W * sum_k a_vec[k]*B[j,k] depends only
        // on the column j — the CPU reference is independent of row sharding.
        std::vector<float> a_vec(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k) a_vec[k] = tk_pattern(static_cast<unsigned long long>(k), 0x1111ULL);
        std::vector<__nv_fp8_e4m3> hA(static_cast<size_t>(M) * K);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                hA[static_cast<size_t>(i) * K + k] = __nv_fp8_e4m3(a_vec[k]);
        std::vector<float> b_host(static_cast<size_t>(N) * K);
        std::vector<__nv_fp8_e4m3> hB(static_cast<size_t>(N) * K);
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k) {
                float bv = tk_pattern(static_cast<unsigned long long>(j) * K + k, 0x2222ULL);
                b_host[static_cast<size_t>(j) * K + k] = bv;
                hB[static_cast<size_t>(j) * K + k] = __nv_fp8_e4m3(bv);
            }
        MatmulReduceScatter::cuda_check(cudaMemcpy(A.ptr(), hA.data(), A_bytes, cudaMemcpyHostToDevice));
        MatmulReduceScatter::cuda_check(cudaMemcpy(B.ptr(), hB.data(), B_bytes, cudaMemcpyHostToDevice));
        // Zero every rank's C shard — `store_add_async` ADDs into it.
        MatmulReduceScatter::cuda_check(cudaMemsetAsync(C.raw_ptrs[comm.rank()], 0, C_bytes));
        MatmulReduceScatter::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, barrier_bytes));
        MatmulReduceScatter::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        run_once(M, K, N, A, B, C, barrier);
        MatmulReduceScatter::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        const size_t shard_rows = static_cast<size_t>(M / W);
        const size_t sample_idxs[3] = {
            0,
            (shard_rows / 2) * N + (N / 2),
            (shard_rows - 1) * N + (N - 1),
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
            MatmulReduceScatter::cuda_check(cudaMemcpy(
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
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_fp8_e4m3);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_fp8_e4m3);
        const size_t C_bytes = static_cast<size_t>(M / W) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = sizeof(int);

        DeviceBuffer A(A_bytes);
        DeviceBuffer B(B_bytes);
        ParallelTensor C(comm.broker(), C_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Non-zero, non-constant inputs so a kernel cannot fake fast timing by
        // detecting all-zero data and skipping the matmul.
        fill_pattern_fp8_kernel<<<dim3((A_bytes / sizeof(__nv_fp8_e4m3) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_fp8_e4m3 *>(A.ptr()), A_bytes / sizeof(__nv_fp8_e4m3), 0x1111ULL);
        fill_pattern_fp8_kernel<<<dim3((B_bytes / sizeof(__nv_fp8_e4m3) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_fp8_e4m3 *>(B.ptr()), B_bytes / sizeof(__nv_fp8_e4m3), 0x2222ULL);
        MatmulReduceScatter::cuda_check(cudaMemsetAsync(C.raw_ptrs[comm.rank()], 0, C_bytes));
        MatmulReduceScatter::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, barrier_bytes));
        MatmulReduceScatter::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            run_once(M, K, N, A, B, C, barrier);
        MatmulReduceScatter::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        MatmulReduceScatter::cuda_check(cudaEventCreate(&start_evt));
        MatmulReduceScatter::cuda_check(cudaEventCreate(&stop_evt));
        MatmulReduceScatter::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            run_once(M, K, N, A, B, C, barrier);
        MatmulReduceScatter::cuda_check(cudaEventRecord(stop_evt));
        MatmulReduceScatter::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        MatmulReduceScatter::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        MatmulReduceScatter::cuda_check(cudaEventDestroy(start_evt));
        MatmulReduceScatter::cuda_check(cudaEventDestroy(stop_evt));

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

    MatmulReduceScatter comm(rank, world_size);

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
    constexpr int WORLD_SIZE = matmul_reduce_scatter::NUM_DEVICES;

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
