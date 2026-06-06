// ThunderKittens BF16 multi-GPU GEMM + AllReduce (NVLink/multicast).
// Empty template: complete the three device kernels marked `// TODO`.
//
// All host scaffolding (per-process fork-and-join, KittensBroker, VMM/IPC
// setup, multicast bring-up, A / B device buffers, correctness check, perf
// benchmark, JSON output) is already wired up — the model only needs to
// fill the device-side `matmul::kernel`, `all_reduce::kernel`, and
// `bracket_barrier::kernel` bodies.
//
// What you need to implement:
//   * `matmul::kernel(g)` — cluster-of-2 Blackwell tcgen05 GEMM
//     `g.C = g.A @ g.B^T` (A is (M, K), B is (N, K), C is (M, N)). Mirror
//     the compute structure of `ag_gemm_b200.cu`'s `comp_sm`: cluster
//     size 2, 1 producer warpgroup (warp 3 lane 0 = TMA loader, cta_rank 0
//     warp 0 lane 0 = `mm2_ABt`/`mma2_ABt` driver) + 1 consumer warpgroup
//     (8-stage epilogue: `warpgroup::load_async` from tensor memory →
//     `warpgroup::store(C_smem)` → `warpgroup::tma::store_async` to g.C).
//   * `all_reduce::kernel(G)` — multimem in-place all-reduce. Compute
//     this thread's offset (`N_per_dev * G.dev_idx + NUM_ELEMS_PER_BLOCK *
//     blockIdx.x + NUM_ELEMS_PER_INST * threadIdx.x`), then call
//     `multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, &G.C.mc_ptr[idx])`
//     and `multimem<bf16_2>::st(&G.C.mc_ptr[idx], tmp)`.
//   * `bracket_barrier::kernel(G)` — one-liner: `barrier_all(G.barrier,
//     {0}, G.dev_idx)`.
//
// Refer to `ref_thunderkitten_gemm_ar.cu` (this directory) and
// `ref_thunderkitten_ag_gemm.cu` (example51) for reference implementations
// of the cluster-of-2 tcgen05 GEMM pattern.

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/broker.cuh"
#include "types/system/vmm.cuh"
#include "types/system/ipc.cuh"

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

using namespace kittens;
using namespace kittens::prototype;


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
};

struct globals {
    static constexpr int PIPELINE_STAGES = 5;
    static constexpr int MMA_PIPE_DEPTH = 2;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<COL_BLOCK / 2, RED_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK / EPI_PIPE_DEPTH>;

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;

    A_gl A;
    B_gl B;
    C_gl C;
};

__device__ inline void kernel(const globals &g) {
    // TODO: cluster-of-2 Blackwell tcgen05 GEMM `g.C = g.A @ g.B^T`.
    //
    // Big-picture pipeline (mirrors `ag_gemm_b200.cu`'s `comp_sm`):
    //
    //   1. Compute scheduling constants:
    //        cta_rank, cluster_idx,
    //        row_blocks = g.A.rows() / ROW_BLOCK,
    //        col_blocks = g.B.rows() / COL_BLOCK   (B is (N, K) so rows=N),
    //        super_rows / final_rows / super_blocks (for SUPER_M tiling),
    //        num_blocks   = row_blocks * col_blocks,
    //        iters_per_task = g.A.cols() / RED_BLOCK.
    //
    //   2. Allocate dynamic shared:
    //        A_smem[PIPELINE_STAGES], B_smem[PIPELINE_STAGES], C_smem[2]
    //      via `tma_swizzle_allocator`. Plus `tensor_allocator<1, 2> tm_alloc`
    //      for tensor memory.
    //
    //   3. Initialize semaphores (only thread 0):
    //        inputs_arrived[PIPELINE_STAGES] @ count 1,
    //        inputs_finished[PIPELINE_STAGES] @ count 1,
    //        outputs_arrived @ count 1,
    //        outputs_finished[MMA_PIPE_DEPTH] @ count CLUSTER_SIZE.
    //      Then `everyone::tma::cluster::sync()`.
    //
    //   4. Branch on `warpgroup::groupid()`:
    //
    //      * groupid == 1 (producers, `increase_registers<256>`):
    //          - lane==0 && warpgroup::warpid()==3 → TMA loader.
    //            Walk task_id = cluster_idx; task_id < num_blocks;
    //                       task_id += C::NUM_BLOCKS / C::CLUSTER_SIZE.
    //            Per task, derive (row_idx, col_idx) using the SUPER_M
    //            tiling (`task_id < super_rows * col_blocks` branch vs.
    //            remainder branch). Then for idx in [0, iters_per_task):
    //                tma::cluster::wait(inputs_finished[input_ring], …)
    //                tma::cluster::load_async(A_smem[input_ring], g.A,
    //                                          {row_idx*2+cta_rank, idx}, …)
    //                tma::cluster::load_async(B_smem[input_ring], g.B,
    //                                          {col_idx*2+cta_rank, idx}, …)
    //                advance input_ring.
    //
    //          - cta_rank==0 && lane==0 && warpgroup::warpid()==0 → MMA driver.
    //            Allocate `d_tt[MMA_PIPE_DEPTH]` from `tm_alloc`.
    //            Per task: wait(outputs_finished[mma_ring], …),
    //                      for idx in [0, iters_per_task):
    //                          tma::cluster::expect_bytes(...)
    //                          tma::cluster::wait(inputs_arrived[input_ring], …)
    //                          if idx==0: mm2_ABt(d_tt[mma_ring], A_smem,
    //                                             B_smem, inputs_finished[input_ring])
    //                          else:      mma2_ABt(...)
    //                      Then `kittens::detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived)`
    //                      and advance mma_ring.
    //
    //      * groupid == 0 (consumer/epilogue, `increase_registers<256>`):
    //          Allocate the same `d_tt[MMA_PIPE_DEPTH]` view.
    //          Per task: wait(outputs_arrived, mma_ring),
    //                    for i in [0, EPI_PIPE_DEPTH):
    //                        warpgroup::load_async(C_reg,
    //                            d_tt[mma_ring].template subtile<...>(0,
    //                                COL_BLOCK/EPI_PIPE_DEPTH*i))
    //                        tensor_load_wait()
    //                        if i == EPI_PIPE_DEPTH-1: warpgroup::sync(1);
    //                            warpgroup::tma::cluster::arrive(outputs_finished[mma_ring], 0)
    //                        warpgroup::tma::store_async_read_wait<1>(); sync(1)
    //                        warpgroup::store(C_smem[i%2], C_reg); sync(1)
    //                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
    //                            g.C, C_smem[i%2],
    //                            {row_idx*2+cta_rank, EPI_PIPE_DEPTH*col_idx+i})
    //                    advance mma_ring.
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
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 2;
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout C;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(C.numel() / NUM_ELEMS_PER_BLOCK / NUM_DEVICES);
    }
};

__device__ inline void kernel(const globals &G) {
    // TODO: in-place multimem all-reduce over the multicast `G.C`.
    //
    //   1. Compute this thread's element index:
    //         N_per_dev = G.C.numel() / NUM_DEVICES
    //         idx = N_per_dev * G.dev_idx
    //             + NUM_ELEMS_PER_BLOCK * blockIdx.x
    //             + NUM_ELEMS_PER_INST  * threadIdx.x
    //   2. Reduce + store through the multicast pointer:
    //         bf16_2 tmp;
    //         multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, &G.C.mc_ptr[idx]);
    //         multimem<bf16_2>::st(&G.C.mc_ptr[idx], tmp);
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
    KittensBroker &broker() { return broker_; }
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
        using mm_globals = matmul::globals;
        using ar_globals = all_reduce::globals;
        using br_globals = bracket_barrier::globals;

        mm_globals mm_g{
            .A = make_gl<mm_globals::A_gl>(reinterpret_cast<uint64_t>(A.ptr()), 1, 1, M, K),
            .B = make_gl<mm_globals::B_gl>(reinterpret_cast<uint64_t>(B.ptr()), 1, 1, N, K),
            .C = make_gl<mm_globals::C_gl>(
                reinterpret_cast<uint64_t>(C.raw_ptrs[rank_]), 1, 1, M, N),
        };

        ar_globals::parallel_layout pgl_C = make_pgl<ar_globals::parallel_layout>(
            reinterpret_cast<uint64_t>(C.mc_ptr),
            reinterpret_cast<uint64_t *>(C.raw_ptrs),
            1, 1, M, N);
        ar_globals ar_g{.C = pgl_C, .dev_idx = rank_};

        using br_barrier_t = barrier_t<br_globals::NUM_DEVICES>;
        br_barrier_t br_pgl = make_pgl<br_barrier_t>(
            reinterpret_cast<uint64_t>(barrier.mc_ptr),
            reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
            1, 1, 1, 1);
        br_globals br_g{.barrier = br_pgl, .dev_idx = rank_};

        // 1. cross-device rendezvous so all ranks have written their A/B
        //    before any rank starts its matmul (avoids reading stale C).
        launch_kernel<bracket_barrier::config, br_globals, bracket_barrier::kernel>(br_g);
        // 2. local matmul into the local shard of C (multicast tensor).
        launch_kernel<matmul::config, mm_globals, matmul::kernel>(mm_g);
        // 3. cross-device rendezvous before the all-reduce.
        launch_kernel<bracket_barrier::config, br_globals, bracket_barrier::kernel>(br_g);
        // 4. in-place multimem all-reduce over the multicast C.
        launch_kernel<all_reduce::config, ar_globals, all_reduce::kernel>(ar_g);
        // 5. final rendezvous so subsequent reads see the reduced value.
        launch_kernel<bracket_barrier::config, br_globals, bracket_barrier::kernel>(br_g);
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
