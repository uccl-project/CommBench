// ThunderKittens FP8 multi-GPU GEMM + ReduceScatter (NVLink/multicast).
// Empty template: complete the two device kernels marked `// TODO`.
//
// All host scaffolding (per-process fork-and-join, KittensBroker, VMM/IPC
// setup, A / B device buffers, correctness check, perf benchmark, JSON
// output) is already wired up — the model only needs to fill the
// device-side `matmul_reduce_scatter::kernel` and
// `matmul_reduce_scatter_barrier::kernel` bodies.
//
// What you need to implement:
//   * `matmul_reduce_scatter::kernel(g)` — cluster-of-2 Blackwell tcgen05
//     `g.d = g.a @ g.b^T` (FP8 inputs, bf16 output) followed by per-tile
//     reduce-scatter to peer ranks. Architecture (mirrors
//     `gemm_rs_fp8_b200.cu`):
//       - 1 producer warpgroup, 2 consumer warpgroups, 4-stage pipeline.
//       - Producers: warp 3 lane 0 = TMA loader (using
//         `warp::tma::cluster::load_async` for FP8 `a`, `b`); cta_rank 0
//         warps 0/1 lane 0 = `mm2_ABt`/`mma2_ABt` MMA driver into a
//         per-warp `tt<float, Mb, Nb>` accumulator.
//       - Consumers: read tensor memory → register tile → bf16 shared
//         `d_smem` → `warp::tma::store_add_async(g.d[dst_dev_idx],
//         d_smem, {row, col})` (only warp 0 of the warpgroup issues),
//         where `dst_dev_idx = rowcol.x / fine_Rblocks_per_dev` selects
//         which peer's shard receives this tile's contribution. The
//         in-network ADD performs reduce-scatter in a single pass.
//     Use the provided `get_task_idx<>` helper for SUPER_M-aware
//     (row, col) scheduling.
//   * `matmul_reduce_scatter_barrier::kernel(G)` — one-liner:
//     `barrier_all(G.barrier, {0}, G.dev_idx)`.
//
// Refer to the upstream `gemm_rs_fp8_b200.cu` (or
// `ref_thunderkitten_gemm_rs_fp8.cu` in this directory) for the
// reference implementation. Note vs. the bf16 variant (`gemm_rs`):
// `Kb = 128` (twice the bf16 reduction tile width); `a` / `b` tiles are
// `st_fp8e4m3`; only warp 0 of each consumer warpgroup issues
// `warp::tma::store_add_async`.
//
// Code layout:
//   * Device-side kernels — `matmul_reduce_scatter::kernel` (cluster-of-2
//     tcgen05 GEMM + per-tile cross-device store-add to the destination
//     rank's C shard) and `matmul_reduce_scatter_barrier::kernel`
//     (multimem.red bracket barrier for end-of-iteration sync).
//   * `ParallelTensor` — pure-C++ replacement for kittens::py::TKParallelTensor.
//   * `DeviceBuffer` — RAII cudaMalloc wrapper for the non-shared A / B.
//   * `MatmulReduceScatter` — clean class. Owns the KittensBroker and
//     exposes only `run` / `sync`.
//   * `runTest(...)` — correctness check + TFLOP/s benchmark.
//   * `printJsonResult(...)` — emits exactly one JSON object on rank 0.
//   * `rank_main` / `main` — fork NUM_DEVICES children and wait.

#include "kittens.cuh"
#include "pyutils/broker.cuh"
#include "types/system/vmm.cuh"
#include "types/system/ipc.cuh"

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

using namespace kittens;


// =====================================================================
//   Device-side kernels — fill in the bodies marked `// TODO`
// =====================================================================

namespace matmul_reduce_scatter {

constexpr int NUM_CONSUMERS = (2);
constexpr int NUM_PRODUCERS = (1);

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 128;   // FP8: doubled reduction tile (vs. 64 for bf16)

static constexpr int NUM_DEVICES = 8;

struct globals {
    using a_tile = st_fp8e4m3<Mb,   Kb>;
    using b_tile = st_fp8e4m3<Nb/2, Kb>;
    using d_tile = st_bf<Mb, 64>;

    using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using d_pgl = pgl<gl<bf16, 1, 1, -1, -1, d_tile>, NUM_DEVICES, false>;

    a_gl a;
    b_gl b;
    d_pgl d;
    const int dev_idx;
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int CLUSTER_M = 4*Mb, CLUSTER_N = Nb;

struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_BLOCKS = 148;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;
};

__device__ static inline int get_iters_per_task(const globals &g) {
    return g.a.cols() / Kb;
}
template<int SUPER_M=8> __device__ static inline int2 get_task_idx(const globals &g, int task_iter, bool is_consumer) {
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    int Rblocks = g.a.rows() / CLUSTER_M, Cblocks = g.b.rows() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    int total_blocks = Rblocks * Cblocks;
    if (task_id < total_blocks) {
        int real_task_id = (task_id + (g.dev_idx + 1) * (total_blocks / NUM_DEVICES)) % total_blocks;
        if (real_task_id < super_rows * Cblocks) {
            return {
                (SUPER_M*(real_task_id/super_repeat) + real_task_id%SUPER_M)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
                is_consumer ? (real_task_id%super_repeat)/SUPER_M : 2*((real_task_id%super_repeat)/SUPER_M) + ctarank
            };
        }
        else {
            int remainder_id = real_task_id - super_rows*Cblocks;
            return {
                (super_rows + remainder_id%final_rows)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
                is_consumer ? remainder_id/final_rows : 2*(remainder_id/final_rows) + ctarank
            };
        }
    }
    else {
        return { -1, -1 };
    }
}

__device__ void kernel(const globals &g) {
    // TODO: cluster-of-2 Blackwell FP8 tcgen05 GEMM (`g.d = g.a @ g.b^T`)
    // followed by per-tile reduce-scatter to peer ranks. Identical
    // structure to `matmul_reduce_scatter::kernel` in the bf16 variant
    // (`ref_thunderkitten_gemm_rs.cu`), with three changes:
    //   1. tile types are `st_fp8e4m3` for `a` / `b` (still `st_bf` for
    //      the staging `d_tile`),
    //   2. `Kb = 128` (twice the bf16 reduction width),
    //   3. only warp 0 of each consumer warpgroup issues the
    //      `warp::tma::store_add_async`.
    //
    // Skeleton:
    //   1. Allocate dynamic shared memory:
    //        a_smem[PIPE_DEPTH][NUM_CONSUMERS]  // (Mb, Kb)  per consumer
    //        b_smem[PIPE_DEPTH]                  // (Nb/2, Kb)
    //        d_smem                              // (Mb, 64) bf16 staging
    //      via `tma_swizzle_allocator`, then `tensor_allocator<1, 2>` for
    //      the per-warp `d_tt = tt<float, Mb, Nb>` accumulator.
    //   2. Initialize semaphores (only thread 0):
    //        inputs_arrived[PIPE_DEPTH]  @ count 2,
    //        inputs_finished[PIPE_DEPTH] @ count NUM_CONSUMERS,
    //        outputs_arrived             @ count 1,
    //        outputs_finished[NUM_CONSUMERS] @ count 2.
    //      Then `everyone::tma::cluster::sync()`.
    //   3. Branch on `warpgroup::groupid()`:
    //      * groupid == NUM_CONSUMERS (producers, decrease_registers<56>):
    //          - warpgroup::warpid()==3 && warp::laneid()==0 → TMA loader.
    //            Walk task_iter; per task, derive (row, col) via
    //            `get_task_idx(g, task_iter, /*is_consumer=*/false)`; end
    //            when row == -1 (drain inputs_finished + arrive on
    //            outputs_arrived once). Per task: for idx in
    //            [0, iters_per_task):
    //                tma::cluster::wait(inputs_finished[input_ring], …)
    //                if appropriate: arrive(outputs_arrived)
    //                warp::tma::cluster::load_async(a_smem[input_ring][0],
    //                                                g.a, {row+0, idx}, …)
    //                warp::tma::cluster::load_async(a_smem[input_ring][1],
    //                                                g.a, {row+1, idx}, …)
    //                warp::tma::cluster::load_async(b_smem[input_ring],
    //                                                g.b, {col, idx}, …)
    //                advance input_ring.
    //          - cta_rank==0 && (warpgroup::warpid()==0 || ==1) &&
    //            warp::laneid()==0 → MMA driver. Allocate
    //            `d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::warpid()*Nb)`.
    //            Per task: wait outputs_finished[warpid] (task_iter+1)%2,
    //                expect+wait inputs_arrived[input_ring], issue
    //                mm2_ABt (first reduction iter) / mma2_ABt (rest) on
    //                d_tt, advance input_ring.
    //      * groupid < NUM_CONSUMERS (consumers, increase_registers<224>):
    //          Allocate `d_tt = tm_alloc.allocate<d_tt_t>(warpgroupid*Nb)`.
    //          Compute fine_Rblocks_per_dev = (g.a.rows()/CLUSTER_M)*4
    //                                         / NUM_DEVICES.
    //          Per task: get_task_idx(g, task_iter, /*is_consumer=*/true),
    //                    derive dst_dev_idx = rowcol.x/fine_Rblocks_per_dev
    //                    and rowcol.x %= fine_Rblocks_per_dev,
    //                    wait(outputs_arrived, task_iter%2),
    //                    load 4 sub-tiles from d_tt into rt_bf<Mb/4, 64>
    //                    via warpgroup::load_async + tensor_load_wait,
    //                    arrive(outputs_finished[warpgroupid]),
    //                    for each sub-tile i:
    //                        warpgroup::store(d_smem, d_reg[i]); sync;
    //                        if (warp 0): warp::tma::store_add_async(
    //                            g.d[dst_dev_idx], d_smem,
    //                            {rowcol.x, 4*rowcol.y+i}).
    //                    `tma::store_async_read_wait()` before reusing d_smem.
    //   4. Final `everyone::tma::cluster::sync()`.
    (void)g;
}

} // namespace matmul_reduce_scatter

namespace matmul_reduce_scatter_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    // TODO: cross-device synchronization. One-liner using kittens::barrier_all
    //       on G.barrier with coord {0} and G.dev_idx.
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
    dim3 grid;
    if constexpr (requires { Config::NUM_BLOCKS; })
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();
    dim3 block = dim3{Config::NUM_THREADS, 1, 1};

    int smem = 0;
    if constexpr (requires { Config::DYNAMIC_SHARED_MEMORY; })
        smem = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    if constexpr (requires { G.dynamic_shared_memory(); })
        smem = G.dynamic_shared_memory();

    if (smem > 0) {
        auto err = cudaFuncSetAttribute(
            (void *)global_kernel<Config, Globals, Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaFuncSetAttribute: ") + cudaGetErrorString(err));
    }

    if constexpr (Config::CLUSTER_SIZE <= 1) {
        global_kernel<Config, Globals, Kernel><<<grid, block, smem, stream>>>(G);
    } else {
        kittens::LaunchConfig<true, false> launch_config(
            grid, block, static_cast<size_t>(smem), stream,
            dim3{Config::CLUSTER_SIZE, 1, 1});
        auto err = cudaLaunchKernelEx(launch_config, global_kernel<Config, Globals, Kernel>, G);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaLaunchKernelEx: ") + cudaGetErrorString(err));
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("kernel launch: ") + cudaGetErrorString(err));
}


// =====================================================================
//   ParallelTensor — pure-C++ replacement for kittens::py::TKParallelTensor.
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

    ParallelTensor(KittensBroker &broker, size_t bytes, int rank, int world_size, bool mc)
        : local_rank(rank), local_world_size(world_size), multicast(mc) {
        if (world_size > MAX_DEVICES)
            throw std::runtime_error("local_world_size > MAX_DEVICES");

        ::kittens::detail::vmm::vm_alloc_map_set_access(
            &raw_ptrs[local_rank], &allocated_size, bytes, local_rank, local_world_size);

        using handle_t = ::kittens::detail::ipc::handle<::kittens::detail::ipc::flavor::VMM>;
        handle_t my_ipc;
        ::kittens::detail::ipc::export_handle(&my_ipc, raw_ptrs[local_rank]);

        std::vector<int> all_fds(local_world_size, -1);
        broker.exchange_fds(all_fds.data(), my_ipc.handle_);

        for (int i = 0; i < local_world_size; ++i) {
            if (i == local_rank) continue;
            handle_t peer;
            peer.handle_ = all_fds[i];
            ::kittens::detail::ipc::import_handle<handle_t>(&raw_ptrs[i], peer, allocated_size, local_world_size);
        }

        if (multicast) initialize_multicast(broker);
    }

    ParallelTensor(const ParallelTensor &) = delete;
    ParallelTensor &operator=(const ParallelTensor &) = delete;
    ~ParallelTensor() = default;

    void initialize_multicast(KittensBroker &broker) {
        using handle_t = ::kittens::detail::ipc::handle<::kittens::detail::ipc::flavor::VMM>;

        ::kittens::detail::vmm::multicast_check(local_rank);
        ::kittens::detail::ipc::check_support(local_rank);
        ::kittens::detail::vmm::handle multicast_handle;

        if (local_rank == 0) {
            ::kittens::detail::vmm::multicast_create_handle(
                &multicast_handle, &mc_allocated_size, allocated_size, local_world_size);
            if (allocated_size != mc_allocated_size)
                throw std::runtime_error("multicast allocated size != memory allocated size");
            handle_t ipc_handle;
            ::kittens::detail::ipc::export_handle(&ipc_handle, multicast_handle);
            broker.broadcast_fd(nullptr, ipc_handle.handle_, 0);
        } else {
            handle_t ipc_handle;
            broker.broadcast_fd(&ipc_handle.handle_, -1, 0);
            mc_allocated_size = allocated_size;
            ::kittens::detail::ipc::import_handle<handle_t>(
                &multicast_handle, ipc_handle, mc_allocated_size, local_world_size);
        }

        ::kittens::detail::vmm::multicast_bind_device(multicast_handle, local_rank);
        broker.sync();

        ::kittens::detail::vmm::handle memory_handle;
        ::kittens::detail::vmm::vm_retrieve_handle(&memory_handle, raw_ptrs[local_rank]);
        ::kittens::detail::vmm::multicast_bind_memory(multicast_handle, memory_handle, allocated_size);
        broker.sync();

        ::kittens::detail::vmm::vm_map(&mc_ptr, multicast_handle, mc_allocated_size);
        ::kittens::detail::vmm::vm_set_access(mc_ptr, mc_allocated_size, local_world_size);

        ::kittens::detail::vmm::vm_free(multicast_handle);
        ::kittens::detail::vmm::vm_free(memory_handle);
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
    KittensBroker &broker() { return broker_; }
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
        matmul_reduce_scatter::globals mm_g{
            .a = make_gl<matmul_reduce_scatter::globals::a_gl>(
                reinterpret_cast<uint64_t>(A.ptr()), 1, 1, M, K),
            .b = make_gl<matmul_reduce_scatter::globals::b_gl>(
                reinterpret_cast<uint64_t>(B.ptr()), 1, 1, N, K),
            .d = make_pgl<matmul_reduce_scatter::globals::d_pgl>(
                reinterpret_cast<uint64_t *>(C.raw_ptrs),
                1, 1, M / world_size_, N),
            .dev_idx = rank_,
        };

        using br_barrier_t = barrier_t<matmul_reduce_scatter_barrier::globals::NUM_DEVICES>;
        br_barrier_t br_pgl = make_pgl<br_barrier_t>(
            reinterpret_cast<uint64_t>(barrier.mc_ptr),
            reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
            1, 1, 1, 1);
        matmul_reduce_scatter_barrier::globals br_g{.barrier = br_pgl, .dev_idx = rank_};

        launch_kernel<matmul_reduce_scatter::config,
                      matmul_reduce_scatter::globals,
                      matmul_reduce_scatter::kernel>(mm_g);
        launch_kernel<matmul_reduce_scatter_barrier::config,
                      matmul_reduce_scatter_barrier::globals,
                      matmul_reduce_scatter_barrier::kernel>(br_g);
    }

    static void cuda_check(cudaError_t err) {
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err));
    }

 private:
    int rank_;
    int world_size_;
    KittensBroker broker_;
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
