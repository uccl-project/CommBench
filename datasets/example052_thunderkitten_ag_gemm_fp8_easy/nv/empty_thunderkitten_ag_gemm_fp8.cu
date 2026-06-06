// ThunderKittens FP8 multi-GPU All-Gather + GEMM (NVLink/multicast).
// Empty template: complete the two device-side helpers marked `// TODO`.
//
// All host scaffolding (per-process fork-and-join, KittensBroker, VMM/IPC
// setup, multicast bring-up, B / C device buffers, correctness check, perf
// benchmark, JSON output) is already wired up — the model only needs to
// fill the device-side `comm_sm` and `comp_sm` bodies.
//
// Inputs `A` (per-rank shard of the activation, gathered via multicast)
// and `B` (replicated weight) are e4m3 FP8; the output `C` is bf16. Note
// vs. the bf16 variant: `RED_BLOCK = 128` (twice as wide) and the tile
// types are `st_fp8e4m3` instead of `st_bf`.
//
// What you need to implement:
//   * `comm_sm(g)` — TMA-driven gather. The first `NUM_CHUNKS` warps each
//     allocate one `A_comm_tile` in shared memory and stream this rank's
//     row shard chunks (`A_smem[warp_id]`) through a 2-stage pipeline:
//     `tma::load_async` from `g.A[g.dev_idx]` → `tma::store_async` through
//     the multicast `g.A`. After the LAST chunk for a given row is stored
//     (when `col_idx + g.num_comm_sms * NUM_CHUNKS >= col_blocks`), call
//     `signal_all(g.barrier, {global_row_idx}, 1)`.
//   * `comp_sm(g)` — cluster-of-2 producer/consumer GEMM. Two warpgroups:
//       - WG1 (warp 3 lane 0): TMA loader. Walks tasks, waits on
//         `g.barrier` for non-local rows, issues `tma::cluster::load_async`
//         of `A_smem` / `B_smem` (FP8 tiles) for `iters_per_task` reduction
//         tiles.
//       - WG1 (warp 0 lane 0, cta_rank 0): MMA driver. Issues
//         `mm2_ABt`/`mma2_ABt` on `d_tt[mma_ring]` (fp32 accumulator) and
//         commits via `kittens::detail::tcgen05::commit`.
//       - WG0: epilogue. After `outputs_arrived`, loads tensor-memory
//         output, stages through `C_smem` (bf16) and `tma::store_async`
//         to `g.C` using an 8-way pipeline.
//
// Refer to the upstream `ag_gemm_fp8_b200.cu` (or
// `ref_thunderkitten_ag_gemm_fp8.cu` in this directory) for the reference
// implementation; the surrounding shared-memory layout, pipeline depths,
// and task scheduling math are already set up to match it.

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/broker.cuh"          // KittensBroker (POSIX SHM + Unix sockets, no torch)
#include "types/system/vmm.cuh"        // CUDA driver VMM (cuMemCreate / cuMulticast*)
#include "types/system/ipc.cuh"        // FD-based IPC handle export/import

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
using namespace kittens::prototype;


// =====================================================================
//   Device-side kernels — fill in the bodies marked `// TODO`
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
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int PIPELINE_STAGES = 5;
    static constexpr int MMA_PIPE_DEPTH = 2;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 128;

    using A_tile = st_fp8e4m3<ROW_BLOCK / 2, RED_BLOCK>;
    using A_comm_tile = st_fp8e4m3<ROW_BLOCK, RED_BLOCK * 2>;
    using B_tile = st_fp8e4m3<COL_BLOCK / 2, RED_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK / EPI_PIPE_DEPTH>;

    static constexpr int NUM_CHUNKS = config::DYNAMIC_SHARED_MEMORY / sizeof(globals::A_comm_tile);

    using A_pgl = pgl<gl<fp8e4m3, 1, 1, -1, -1, A_tile, A_comm_tile>, NUM_DEVICES, true, A_comm_tile>;
    using B_gl = gl<fp8e4m3, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;
    using barrier_pgl = barrier_t<NUM_DEVICES>;

    A_pgl A;
    B_gl B;
    C_gl C;
    barrier_pgl barrier;

    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;
};

__device__ inline void comm_sm(const globals &g) {
    // TODO: communication SM body. Outline:
    //
    //   1. Allocate `globals::NUM_CHUNKS` `A_comm_tile`s (FP8) in dynamic
    //      shared memory via `tma_swizzle_allocator`, plus a parallel
    //      `__shared__ kittens::semaphore inputs_arrived[NUM_CHUNKS]`.
    //   2. Compute scheduling indices:
    //         comm_sm_id        = blockIdx.x - g.num_comp_sms
    //         warp_id           = warp::groupid()
    //         lane_id           = warp::laneid()
    //         global_row_blocks = g.A.rows() / ROW_BLOCK
    //         local_row_blocks  = global_row_blocks / NUM_DEVICES
    //         col_blocks        = g.A.cols() / (RED_BLOCK * 2)
    //         num_local_blocks  = local_row_blocks * col_blocks
    //      Plus a `phasebits = 0xFFFF0000` for `get_phasebit<0>` /
    //      `update_phasebit<0>`.
    //   3. Only the first NUM_CHUNKS warps with lane_id==0 do work.
    //      `init_semaphore(inputs_arrived[warp_id], 0, 1)` once.
    //   4. Walk tasks `task_id = comm_sm_id*NUM_CHUNKS + warp_id`,
    //      stepping by `g.num_comm_sms * NUM_CHUNKS`, until
    //      task_id >= num_local_blocks. Per task:
    //         row_idx        = task_id / col_blocks
    //         global_row_idx = row_idx + g.dev_idx * local_row_blocks
    //         col_idx        = task_id % col_blocks
    //         tma::expect_bytes(inputs_arrived[warp_id], sizeof(A_comm_tile))
    //         tma::load_async(A_smem[warp_id], g.A[g.dev_idx],
    //                         {global_row_idx, col_idx}, inputs_arrived[warp_id])
    //         wait(...) on phasebit<0>; update_phasebit<0>
    //         tma::store_async(g.A, A_smem[warp_id], {global_row_idx, col_idx})
    //         tma::store_async_wait()
    //         if (col_idx + g.num_comm_sms * NUM_CHUNKS >= col_blocks)
    //             signal_all(g.barrier, {global_row_idx}, 1)
}

__device__ inline void comp_sm(const globals &g) {
    // TODO: computation SM body — cluster-of-2 producer/consumer FP8 GEMM.
    //
    // Big-picture pipeline:
    //   * tensor-memory accumulators `d_tt[MMA_PIPE_DEPTH]` (fp32) are
    //     double-buffered across tasks; per task we issue
    //     `iters_per_task = g.A.cols() / RED_BLOCK` mm2/mma2 reduction
    //     tiles. RED_BLOCK is 128 (vs. 64 for bf16) — twice as wide.
    //   * shared-memory ring buffers `A_smem[PIPELINE_STAGES]` /
    //     `B_smem[PIPELINE_STAGES]` (FP8 tiles) are filled by a TMA loader
    //     and consumed by the MMA driver, with `inputs_arrived` /
    //     `inputs_finished` semaphores synchronizing the two.
    //   * the consumer warpgroup reads the tensor-memory output via 8-way
    //     pipelined `warpgroup::load_async` → `warpgroup::store(C_smem)`
    //     (bf16) → `warpgroup::tma::store_async(g.C, ...)`.
    //
    // Skeleton (identical structure to the bf16 variant — only the tile
    // dtypes change):
    //   1. Compute scheduling constants:
    //        cta_rank, cluster_idx, iters_per_task,
    //        global_row_blocks, local_row_blocks, col_blocks (= g.B.rows()/COL_BLOCK!),
    //        super_rows, final_rows, super_blocks,
    //        num_global_blocks, num_local_blocks,
    //        num_comm_workers_per_stage = min(g.num_comm_sms*NUM_CHUNKS,
    //                                          iters_per_task/2)
    //   2. Allocate `A_smem[PIPELINE_STAGES]`, `B_smem[PIPELINE_STAGES]`
    //      (both `st_fp8e4m3`), `C_smem[2]` (`st_bf`) in dynamic shared
    //      memory with `tma_swizzle_allocator`, and
    //      `tensor_allocator<1, 2> tm_alloc` for tensor memory.
    //   3. Initialize semaphores (only thread 0):
    //        inputs_arrived[PIPELINE_STAGES] @ count 1
    //        inputs_finished[PIPELINE_STAGES] @ count 1
    //        outputs_arrived @ count 1
    //        outputs_finished[MMA_PIPE_DEPTH] @ count CLUSTER_SIZE
    //      Then `everyone::tma::cluster::sync()`.
    //   4. Branch on `warpgroup::groupid()`:
    //
    //      * groupid == 1 (producers, increase_registers<256>):
    //          - lane==0 && warpgroup::warpid()==3 → TMA loader thread.
    //            For each task_id = cluster_idx; task_id < num_global_blocks;
    //                            task_id += g.num_comp_sms/2:
    //                derive (row_idx, col_idx) using the SUPER_M-tiled
    //                schedule for task_id < num_local_blocks, otherwise
    //                the peer-shard schedule and call
    //                `wait(g.barrier, {row_idx}, g.dev_idx,
    //                      num_comm_workers_per_stage)`.
    //                Then for idx in [0, iters_per_task):
    //                    tma::cluster::wait(inputs_finished[input_ring], ...)
    //                    update_phasebit<1>(...)
    //                    tma::cluster::load_async(A_smem[input_ring],
    //                                             g.A[g.dev_idx],
    //                                             {row_idx*2+cta_rank, idx}, ...)
    //                    tma::cluster::load_async(B_smem[input_ring], g.B,
    //                                             {col_idx*2+cta_rank, idx}, ...)
    //                    input_ring = ring_advance<PIPELINE_STAGES>(input_ring)
    //
    //          - cta_rank==0 && lane==0 && warpgroup::warpid()==0 → MMA driver.
    //            For each task: wait outputs_finished[mma_ring], then for
    //                idx in [0, iters_per_task):
    //                    tma::cluster::expect_bytes(inputs_arrived[input_ring],
    //                                               2*sizeof(A_tile)+2*sizeof(B_tile))
    //                    tma::cluster::wait(inputs_arrived[input_ring], ...)
    //                    if idx==0: mm2_ABt(d_tt[mma_ring], A_smem, B_smem,
    //                                       inputs_finished[input_ring])
    //                    else:      mma2_ABt(...)
    //                    advance input_ring
    //                Then `kittens::detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived)`
    //                and advance mma_ring.
    //
    //      * groupid == 0 (consumer/epilogue, increase_registers<256>):
    //          For each task_id = cluster_idx; task_id < num_global_blocks;
    //                          task_id += g.num_comp_sms/2:
    //              recompute (row_idx, col_idx) (no barrier wait here)
    //              wait(outputs_arrived, mma_ring)
    //              For i in [0, EPI_PIPE_DEPTH):
    //                  warpgroup::load_async(C_reg, d_tt[mma_ring].template
    //                                        subtile<tt<float, ROW_BLOCK/2,
    //                                        COL_BLOCK/EPI_PIPE_DEPTH>>(0,
    //                                        COL_BLOCK/EPI_PIPE_DEPTH*i))
    //                  tensor_load_wait()
    //                  if i == EPI_PIPE_DEPTH-1: warpgroup::sync(1);
    //                      warpgroup::tma::cluster::arrive(outputs_finished[mma_ring], 0)
    //                  warpgroup::tma::store_async_read_wait<1>(); sync(1)
    //                  warpgroup::store(C_smem[i%2], C_reg); sync(1)
    //                  warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
    //                      g.C, C_smem[i%2],
    //                      {row_idx*2+cta_rank, EPI_PIPE_DEPTH*col_idx+i})
    //              advance mma_ring
}

__device__ inline void main_kernel(const globals &g) {
    if (blockIdx.x < g.num_comp_sms)
        comp_sm(g);
    else
        comm_sm(g);
}

__device__ inline void epilogue_kernel(const globals &g) {
    const int num_blocks = g.A.rows() / globals::ROW_BLOCK;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride)
        g.barrier[g.dev_idx][{i}] = 0;

    if (blockIdx.x == 0 && threadIdx.x == 0)
        barrier_all(g.barrier, {1, 0, 0}, g.dev_idx);
}


// =====================================================================
//   Minimal launch helper. Supports CLUSTER_SIZE >= 1 (uses
//   cudaLaunchKernelEx with kittens::LaunchConfig for clusters).
// =====================================================================

template <typename C>
consteval int kernel_min_blocks_per_sm() {
    if constexpr (requires { C::MIN_BLOCKS_PER_SM; })
        return static_cast<int>(C::MIN_BLOCKS_PER_SM);
    else
        return 1;  // Match upstream `kittens::py::launch_kernel` default.
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
//   ParallelTensor — pure-C++ replacement for kittens::py::TKParallelTensor
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
//   AllGatherMatmul — clean communication class (no test/benchmark logic).
//
//   Owns the KittensBroker and exposes only the core ops:
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
    KittensBroker &broker() { return broker_; }

    void sync() { broker_.sync(); }

    // One bracketed all-gather + matmul launch.
    //
    //   * `A`        : (M, K) fp8e4m3, multicast=true (each rank owns the
    //                  row shard [rank*M/W, (rank+1)*M/W); kernel gathers
    //                  the rest in-place via TMA over the multicast handle).
    //   * `B`        : (N, K) fp8e4m3, plain device buffer, fully replicated
    //                  on every rank (kernel computes A @ B^T in fp32 accum).
    //   * `C`        : (M, N) bf16, plain device buffer, written locally.
    //   * `barrier`  : (2, 1024, 1024) int multicast, cleared by epilogue.
    void run(ParallelTensor &A, DeviceBuffer &B, DeviceBuffer &C,
             ParallelTensor &barrier, int M, int K, int N, int num_comm_sms) {
        using A_pgl = globals::A_pgl;
        using B_gl  = globals::B_gl;
        using C_gl  = globals::C_gl;
        using bar_pgl = globals::barrier_pgl;

        A_pgl pgl_A = make_pgl<A_pgl>(
            reinterpret_cast<uint64_t>(A.mc_ptr),
            reinterpret_cast<uint64_t *>(A.raw_ptrs),
            1, 1, M, K);
        B_gl gl_B = make_gl<B_gl>(
            reinterpret_cast<uint64_t>(B.ptr()), 1, 1, N, K);
        C_gl gl_C = make_gl<C_gl>(
            reinterpret_cast<uint64_t>(C.ptr()), 1, 1, M, N);
        bar_pgl pgl_bar = make_pgl<bar_pgl>(
            reinterpret_cast<uint64_t>(barrier.mc_ptr),
            reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
            1, 2, 1024, 1024);

        globals g{
            .A = pgl_A,
            .B = gl_B,
            .C = gl_C,
            .barrier = pgl_bar,
            .dev_idx = rank_,
            .num_comm_sms = num_comm_sms,
            .num_comp_sms = config::NUM_BLOCKS - num_comm_sms,
        };

        launch_kernel<config, globals, main_kernel>(g);
        launch_kernel<config, globals, epilogue_kernel>(g);
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
//   Test harness (correctness + performance benchmark)
// =====================================================================

struct MetricRow {
    double data_size_mb;       // input A footprint in MB (M*K*1/1e6, fp8)
    double throughput_tflops;  // 2*M*N*K / time, in TFLOP/s
    double latency_ms;
};

// Each entry is (M=K=size, N=size/world_size). Extends the upstream
// `benchmark.py` size sweep with two extra power-of-2 points (3072 and
// 6144 are added so the table has 7 distinct data sizes) — `num_comm_sms`
// is fixed at `kNumCommSms` instead of the upstream 6-value sweep, so the
// example finishes in roughly a minute on B300.
static constexpr int kBenchmarkSizes[] = {2048, 3072, 4096, 6144, 8192, 16384, 32768};
static constexpr int kNumCommSms      = 16;
static constexpr int kNumWarmupIters  = 1;
static constexpr int kNumIters        = 5;
static constexpr int kCorrectnessSize = 2048;  // smallest that satisfies kernel block constraints


// FP8 e4m3 raw byte for `2^r` (r >= 0): exponent biased by 7, mantissa 0.
//   r=0 → 1.0 = 0x38, r=1 → 2.0 = 0x40, …, r=7 → 128.0 = 0x70.
// All values are exactly representable in e4m3 (max normal magnitude 448).
static __host__ inline uint8_t fp8e4m3_byte_pow2(int r) {
    return static_cast<uint8_t>(0x38 + 8 * r);
}


// Standalone test/benchmark function — handles correctness + timing.
//
//   * Correctness: each rank fills its A_local shard with constant 2^rank
//     (an exactly-representable e4m3 value) and B with 1.0. After the
//     all-gather + GEMM, on every rank
//        C[i, j] = sum_k A_full[i, k] * B[j, k] = K * 2^{i_block}
//     where i_block = i / (M/W). All expected values are powers of two
//     that fit exactly in bf16 for K=2048, so we can use a tight 1.0
//     absolute tolerance. We sample C[r * (M/W), 0] for every r ∈ [0, W).
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

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int M = correctness_size;
        const int K = correctness_size;
        const int N = correctness_size / W;
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_fp8_e4m3);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_fp8_e4m3);
        const size_t C_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = 2u * 1024u * 1024u * sizeof(int);

        ParallelTensor A(comm.broker(), A_bytes, comm.rank(), W, true);
        DeviceBuffer B(B_bytes);
        DeviceBuffer C(C_bytes);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Seeded, non-constant, non-zero fp8 inputs. Each rank fills its A row
        // shard [r*M/W, (r+1)*M/W) with a GLOBAL-row-indexed pattern, and B
        // (replicated) with another pattern. After all-gather + GEMM,
        //   C[i,j] = sum_k A_full[i,k] * B[j,k]
        // is fully data-dependent, so a kernel cannot pass by hardcoding or by
        // detecting trivial (all-zero / all-equal) inputs.
        const int shard_rows = M / W;
        const size_t shard_offset_bytes =
            static_cast<size_t>(comm.rank()) * shard_rows * K * sizeof(__nv_fp8_e4m3);
        std::vector<__nv_fp8_e4m3> hA(static_cast<size_t>(shard_rows) * K);
        for (int lr = 0; lr < shard_rows; ++lr) {
            int grow = comm.rank() * shard_rows + lr;
            for (int k = 0; k < K; ++k)
                hA[static_cast<size_t>(lr) * K + k] =
                    __nv_fp8_e4m3(tk_pattern(static_cast<unsigned long long>(grow) * K + k, 0x1111ULL));
        }
        AllGatherMatmul::cuda_check(cudaMemcpy(
            reinterpret_cast<uint8_t *>(A.raw_ptrs[comm.rank()]) + shard_offset_bytes,
            hA.data(), static_cast<size_t>(shard_rows) * K * sizeof(__nv_fp8_e4m3),
            cudaMemcpyHostToDevice));
        std::vector<float> b_host(static_cast<size_t>(N) * K);
        std::vector<__nv_fp8_e4m3> hB(static_cast<size_t>(N) * K);
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k) {
                float bv = tk_pattern(static_cast<unsigned long long>(j) * K + k, 0x2222ULL);
                b_host[static_cast<size_t>(j) * K + k] = bv;
                hB[static_cast<size_t>(j) * K + k] = __nv_fp8_e4m3(bv);
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
        const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(__nv_fp8_e4m3);
        const size_t B_bytes = static_cast<size_t>(N) * K * sizeof(__nv_fp8_e4m3);
        const size_t C_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
        const size_t barrier_bytes = 2u * 1024u * 1024u * sizeof(int);

        ParallelTensor A(comm.broker(), A_bytes, comm.rank(), W, true);
        DeviceBuffer B(B_bytes);
        DeviceBuffer C(C_bytes);
        ParallelTensor barrier(comm.broker(), barrier_bytes, comm.rank(), W, true);

        // Non-zero, non-constant inputs so a kernel cannot fake fast timing by
        // detecting all-zero data and skipping the all-gather + matmul.
        fill_pattern_fp8_kernel<<<dim3((A_bytes / sizeof(__nv_fp8_e4m3) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_fp8_e4m3 *>(A.raw_ptrs[comm.rank()]), A_bytes / sizeof(__nv_fp8_e4m3), 0x1111ULL);
        fill_pattern_fp8_kernel<<<dim3((B_bytes / sizeof(__nv_fp8_e4m3) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_fp8_e4m3 *>(B.ptr()), B_bytes / sizeof(__nv_fp8_e4m3), 0x2222ULL);
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
