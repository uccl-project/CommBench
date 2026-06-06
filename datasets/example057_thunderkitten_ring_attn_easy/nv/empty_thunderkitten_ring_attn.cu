// ThunderKittens BF16 multi-GPU Ring-Attention K/V rotation (NVLink/IPC).
// Empty template: complete the two device kernels marked `// TODO` below.
//
// All host scaffolding (per-process fork-and-join, KittensBroker, VMM/IPC
// setup, multicast bring-up, ParallelTensor, RingAttnComm, correctness
// check, perf benchmark, JSON output) is already wired up — the model
// only needs to fill the device-side bodies.
//
// What you need to implement:
//   * `attn_comm::kernel` — TMA-driven cross-rank K/V ring rotation. The
//     block is split across `NUM_CHUNKS = 7` producer warps and 7 consumer
//     warps that share `NUM_CHUNKS` smem tiles via `inputs_arrived` /
//     `inputs_finished` mbarriers. Even-indexed blocks rotate K, odd
//     blocks rotate V. On stage `s`, the source buffer is K{s%2}/V{s%2}
//     on this rank and the destination is K{(s+1)%2}/V{(s+1)%2} on rank
//     `(dev_idx+1) mod NUM_DEVICES`.
//   * `barrier_ns::kernel` — synchronize all NUM_DEVICES ranks via
//     `barrier_all` on G.barrier.

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
//   Device-side kernels — fill in the bodies marked `// TODO`
// =====================================================================

namespace attn_comm {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - STATIC_SHARED_MEMORY;
    // 4 warpgroups × 4 warps × 32 threads = 512 threads (matches upstream's
    // attn_comm_partial_kernel grid; only ~14 warps actually do work).
    static constexpr int NUM_THREADS = 4 * WARPGROUP_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;

    static constexpr int D = 128;
    static constexpr int KV_BLOCK = 128;

    using K_tile = st_bf<KV_BLOCK, D>;
    using V_tile = st_bf<KV_BLOCK, D>;

    using K_pgl = pgl<gl<bf16, -1, -1, -1, D, K_tile>, NUM_DEVICES, false>;
    using V_pgl = pgl<gl<bf16, -1, -1, -1, D, V_tile>, NUM_DEVICES, false>;
    using barrier_pgl = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, true>;

    K_pgl K0;
    K_pgl K1;
    V_pgl V0;
    V_pgl V1;

    barrier_pgl barrier;

    int ring_stage;
    const int dev_idx;
    const int num_comm_sms;  // must be even
    const int num_K_blocks_per_dev;
    const int num_heads;
    const int num_batches;
};

__device__ inline void kernel(const globals &G) {
    // TODO: TMA-driven cross-rank K/V ring rotation.
    //
    // 1. Allocate `NUM_CHUNKS = 7` smem tiles of `globals::K_tile` from
    //    dynamic shared memory using `tma_swizzle_allocator` over
    //    `extern __shared__ int __shm[]`. Static-assert that they fit in
    //    `config::DYNAMIC_SHARED_MEMORY`.
    // 2. Compute:
    //      - block_idx     = blockIdx.x       (even => send K, odd => send V)
    //      - warp_id       = warp::groupid()
    //      - num_blocks    = G.num_batches * G.num_heads * G.num_K_blocks_per_dev
    //      - dst_dev_idx   = (G.dev_idx + 1) % NUM_DEVICES
    // 3. Initialize two arrays of `__shared__ kittens::semaphore`:
    //      `inputs_arrived[NUM_CHUNKS]`, `inputs_finished[NUM_CHUNKS]`
    //    (thread 0 calls `init_semaphore(.., 0, 1)`); then `__syncthreads()`.
    //    Initialize `uint32_t phasebits = 0xFFFF0000`.
    // 4. Producer (warp_id < NUM_CHUNKS, laneid() == 0): for
    //    `task_id = NUM_CHUNKS * (block_idx / 2) + chunk_id` stepped by
    //    `NUM_CHUNKS * (G.num_comm_sms / 2)` over [0, num_blocks):
    //      - decode {batch_idx, head_idx, KV_idx} from task_id
    //      - wait(inputs_finished[chunk_id], get_phasebit<1>(phasebits, 0))
    //        and update_phasebit<1>(phasebits, 0)
    //      - tma::expect_bytes(inputs_arrived[chunk_id], sizeof(K_tile))
    //      - tma::load_async into KV_smem[chunk_id] from
    //          G.{K|V}{ring_stage%2}[G.dev_idx]
    //        at coords {batch_idx, head_idx, KV_idx, 0}, signaling
    //        `inputs_arrived[chunk_id]`. K vs V is selected by
    //        `block_idx % 2`.
    // 5. Consumer (NUM_CHUNKS <= warp_id < 2*NUM_CHUNKS, laneid() == 0):
    //    same task_id loop with chunk_id = warp_id - NUM_CHUNKS:
    //      - wait(inputs_arrived[chunk_id], get_phasebit<0>(phasebits, 0))
    //        and update_phasebit<0>(phasebits, 0)
    //      - tma::store_async to
    //          G.{K|V}{(ring_stage+1)%2}[dst_dev_idx]
    //        from KV_smem[chunk_id] at coords {batch_idx, head_idx, KV_idx, 0}
    //      - tma::store_async_read_wait()
    //      - arrive(inputs_finished[chunk_id])
}

} // namespace attn_comm


// =====================================================================
//   Cross-device barrier kernel.
// =====================================================================

namespace barrier_ns {

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
    //       on G.barrier with coord {1, 0, 0} and G.dev_idx.
}

} // namespace barrier_ns


// =====================================================================
//   Launch helper.
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
__host__ inline void launch_kernel(dim3 grid, const Globals &G,
                                   cudaStream_t stream = 0) {
    dim3 block = dim3{Config::NUM_THREADS, 1, 1};
    int smem = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    if (smem > 0) {
        auto err = cudaFuncSetAttribute(
            (void *)global_kernel<Config, Globals, Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaFuncSetAttribute: ")
                                     + cudaGetErrorString(err));
    }
    global_kernel<Config, Globals, Kernel><<<grid, block, smem, stream>>>(G);
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
//   Init kernel for the correctness check.
// =====================================================================

__global__ void fill_bf16_kernel(__nv_bfloat16 *p, size_t n, __nv_bfloat16 v) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// Deterministic, non-constant, non-zero test pattern (values in {0.5,1.0,1.5,2.0},
// all bf16-exact so a round-trip copy is lossless). A per-element pattern forces
// the ring rotation to actually preserve real data — a kernel that writes a
// constant or garbage, or that detects all-zero input and skips work, fails.
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
//   RingAttnComm — clean communication class.
// =====================================================================

class RingAttnComm {
 public:
    RingAttnComm(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    RingAttnComm(const RingAttnComm &) = delete;
    RingAttnComm &operator=(const RingAttnComm &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    KittensBroker &broker() { return broker_; }
    void sync() { broker_.sync(); }

    // Run one ring stage: TMA-rotate K and V from `K{ring_stage%2}` /
    // `V{ring_stage%2}` on this rank to `K{(ring_stage+1)%2}` /
    // `V{(ring_stage+1)%2}` on rank `(rank+1) mod world_size`. Followed
    // by a cross-device barrier.
    void run_one_stage(ParallelTensor &K0, ParallelTensor &K1,
                       ParallelTensor &V0, ParallelTensor &V1,
                       ParallelTensor &barrier,
                       int B, int H, int N_per_dev, int ring_stage,
                       int num_comm_sms) {
        const int D = attn_comm::globals::D;
        const int KV_BLOCK = attn_comm::globals::KV_BLOCK;

        attn_comm::globals::K_pgl K0_pgl = make_pgl<attn_comm::globals::K_pgl>(
            reinterpret_cast<uint64_t *>(K0.raw_ptrs), B, H, N_per_dev, D);
        attn_comm::globals::K_pgl K1_pgl = make_pgl<attn_comm::globals::K_pgl>(
            reinterpret_cast<uint64_t *>(K1.raw_ptrs), B, H, N_per_dev, D);
        attn_comm::globals::V_pgl V0_pgl = make_pgl<attn_comm::globals::V_pgl>(
            reinterpret_cast<uint64_t *>(V0.raw_ptrs), B, H, N_per_dev, D);
        attn_comm::globals::V_pgl V1_pgl = make_pgl<attn_comm::globals::V_pgl>(
            reinterpret_cast<uint64_t *>(V1.raw_ptrs), B, H, N_per_dev, D);

        attn_comm::globals::barrier_pgl br_pgl =
            make_pgl<attn_comm::globals::barrier_pgl>(
                reinterpret_cast<uint64_t>(barrier.mc_ptr),
                reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
                1, 2, 1024, 1024);

        attn_comm::globals g{
            .K0 = K0_pgl,
            .K1 = K1_pgl,
            .V0 = V0_pgl,
            .V1 = V1_pgl,
            .barrier = br_pgl,
            .ring_stage = ring_stage,
            .dev_idx = rank_,
            .num_comm_sms = num_comm_sms,
            .num_K_blocks_per_dev = N_per_dev / KV_BLOCK,
            .num_heads = H,
            .num_batches = B,
        };

        launch_kernel<attn_comm::config, attn_comm::globals, attn_comm::kernel>(
            dim3(num_comm_sms, 1, 1), g);

        barrier_ns::globals br_g{.barrier = br_pgl, .dev_idx = rank_};
        launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(
            dim3(barrier_ns::config::NUM_BLOCKS, 1, 1), br_g);
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
//   Test harness (correctness + bandwidth benchmark).
// =====================================================================

struct MetricRow {
    double data_size_mb;       // total K+V bytes rotated per rank per *full ring*
    double throughput_gbps;    // (bytes per stage * num_stages) / time
    double latency_ms;         // total time for a full ring (NUM_DEVICES stages)
};

// Fixed model dims (kept small so the example finishes in well under a minute
// on B300).
static constexpr int kBatch  = 2;
static constexpr int kHeads  = 4;
static constexpr int kDim    = 128;     // matches attn_comm::globals::D
static constexpr int kBenchmarkN_per_dev[] = {256, 512, 1024, 2048, 4096, 8192};
static constexpr int kNumCommSms = 16;   // must be even
static constexpr int kNumWarmup  = 1;
static constexpr int kNumIters   = 5;
static constexpr int kCorrectN_per_dev = 256;

// Correctness: each rank fills K0 / V0 with constant `(rank+1) * 0.001`.
// One full ring (NUM_DEVICES stages) advances data through every peer and
// returns it home, so after stage NUM_DEVICES-1 the K0 / V0 buffers
// (alternating with K1 / V1 across stages, see the kernel comment) hold
// the original `(rank+1)*0.001` value again.
//
// In particular: after ring_stage = NUM_DEVICES - 1 = 7 (odd), the consumer
// stores into `K0` (because ring_stage%2 == 1 chooses K0 as the dst), so
// rank r's `K0` now holds the data that originated on rank r itself.
static std::pair<bool, std::vector<MetricRow>> runTest(
    RingAttnComm &comm,
    const int *N_sizes, int num_sizes,
    int correctness_N,
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

    auto run_full_ring = [&](ParallelTensor &K0, ParallelTensor &K1,
                             ParallelTensor &V0, ParallelTensor &V1,
                             ParallelTensor &barrier,
                             int B, int H, int N_per_dev) {
        for (int s = 0; s < W; ++s)
            comm.run_one_stage(K0, K1, V0, V1, barrier, B, H, N_per_dev, s, kNumCommSms);
    };

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int B = kBatch;
        const int H = kHeads;
        const int N_per_dev = correctness_N;
        const size_t kv_bytes = static_cast<size_t>(B) * H * N_per_dev * kDim
                              * sizeof(__nv_bfloat16);
        const size_t bar_bytes = 2u * 1024u * 1024u * sizeof(int);

        ParallelTensor K0(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor K1(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor V0(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor V1(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), bar_bytes, comm.rank(), W, true);

        // Seeded, non-constant, non-zero per-element inputs (distinct per rank).
        // bf16-exact values round-trip losslessly, so after the full ring K0/V0
        // must hold their original per-element pattern — a kernel that fills a
        // constant, writes garbage, or skips work on detecting all-zero fails.
        const unsigned long long seedK = 0x100ULL + static_cast<unsigned long long>(comm.rank());
        const unsigned long long seedV = 0x200ULL + static_cast<unsigned long long>(comm.rank());
        const size_t kv_numel = kv_bytes / sizeof(__nv_bfloat16);
        fill_pattern_kernel<<<dim3((kv_numel + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(K0.raw_ptrs[comm.rank()]), kv_numel, seedK);
        fill_pattern_kernel<<<dim3((kv_numel + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(V0.raw_ptrs[comm.rank()]), kv_numel, seedV);
        RingAttnComm::cuda_check(cudaMemsetAsync(K1.raw_ptrs[comm.rank()], 0, kv_bytes));
        RingAttnComm::cuda_check(cudaMemsetAsync(V1.raw_ptrs[comm.rank()], 0, kv_bytes));
        RingAttnComm::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, bar_bytes));
        RingAttnComm::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        run_full_ring(K0, K1, V0, V1, barrier, B, H, N_per_dev);
        RingAttnComm::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        // After NUM_DEVICES = 8 stages, K0 / V0 should hold the original
        // per-element values (data has rotated all the way around).
        const size_t sample_idxs[3] = {
            0,
            (kv_bytes / sizeof(__nv_bfloat16)) / 2,
            (kv_bytes / sizeof(__nv_bfloat16)) - 1,
        };
        __nv_bfloat16 host_K[3]{}, host_V[3]{};
        for (int i = 0; i < 3; ++i) {
            RingAttnComm::cuda_check(cudaMemcpy(
                &host_K[i],
                reinterpret_cast<__nv_bfloat16 *>(K0.raw_ptrs[comm.rank()]) + sample_idxs[i],
                sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));
            RingAttnComm::cuda_check(cudaMemcpy(
                &host_V[i],
                reinterpret_cast<__nv_bfloat16 *>(V0.raw_ptrs[comm.rank()]) + sample_idxs[i],
                sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));
        }
        overall_pass = true;
        for (int i = 0; i < 3; ++i) {
            float vk = __bfloat162float(host_K[i]);
            float vv = __bfloat162float(host_V[i]);
            float wantK = tk_pattern(sample_idxs[i], seedK);
            float wantV = tk_pattern(sample_idxs[i], seedV);
            if (std::fabs(vk - wantK) > 1e-2f ||
                std::fabs(vv - wantV) > 1e-2f) {
                overall_pass = false;
                break;
            }
        }
        comm.sync();
    }

    // ---- benchmark ----
    for (int s = 0; s < num_sizes; ++s) {
        const int B = kBatch;
        const int H = kHeads;
        const int N_per_dev = N_sizes[s];
        const size_t kv_bytes = static_cast<size_t>(B) * H * N_per_dev * kDim
                              * sizeof(__nv_bfloat16);
        const size_t bar_bytes = 2u * 1024u * 1024u * sizeof(int);

        ParallelTensor K0(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor K1(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor V0(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor V1(comm.broker(), kv_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), bar_bytes, comm.rank(), W, true);

        // Non-zero, non-constant inputs so a kernel cannot fake fast timing by
        // detecting all-zero KV and skipping the ring rotation.
        {
            const size_t kv_numel = kv_bytes / sizeof(__nv_bfloat16);
            fill_pattern_kernel<<<dim3((kv_numel + 255) / 256), 256>>>(
                reinterpret_cast<__nv_bfloat16 *>(K0.raw_ptrs[comm.rank()]), kv_numel,
                0x100ULL + static_cast<unsigned long long>(comm.rank()));
            fill_pattern_kernel<<<dim3((kv_numel + 255) / 256), 256>>>(
                reinterpret_cast<__nv_bfloat16 *>(V0.raw_ptrs[comm.rank()]), kv_numel,
                0x200ULL + static_cast<unsigned long long>(comm.rank()));
        }
        RingAttnComm::cuda_check(cudaMemsetAsync(K1.raw_ptrs[comm.rank()], 0, kv_bytes));
        RingAttnComm::cuda_check(cudaMemsetAsync(V1.raw_ptrs[comm.rank()], 0, kv_bytes));
        RingAttnComm::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, bar_bytes));
        RingAttnComm::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            run_full_ring(K0, K1, V0, V1, barrier, B, H, N_per_dev);
        RingAttnComm::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        RingAttnComm::cuda_check(cudaEventCreate(&start_evt));
        RingAttnComm::cuda_check(cudaEventCreate(&stop_evt));
        RingAttnComm::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            run_full_ring(K0, K1, V0, V1, barrier, B, H, N_per_dev);
        RingAttnComm::cuda_check(cudaEventRecord(stop_evt));
        RingAttnComm::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        RingAttnComm::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        RingAttnComm::cuda_check(cudaEventDestroy(start_evt));
        RingAttnComm::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms_per_ring = total_ms / iters;
        // Bytes moved out of this rank in one full ring:
        //   2 (K + V) tensors × NUM_DEVICES stages × kv_bytes per stage.
        const double bytes_per_ring =
            2.0 * static_cast<double>(W) * static_cast<double>(kv_bytes);
        const double gbps = avg_ms_per_ring > 0.0
            ? (bytes_per_ring * 1e-9) / (avg_ms_per_ring * 1e-3)
            : 0.0;

        rows.push_back({
            bytes_per_ring / 1e6,
            gbps,
            avg_ms_per_ring,
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
    std::cout << "  \"throughput_unit\": \"GB/s\",\n";
    std::cout << "  \"latency_unit\": \"ms\",\n";
    std::cout << "  \"metrics\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto &r = rows[i];
        std::cout << "    {\"data_size\": " << r.data_size_mb
                  << ", \"throughput_avg\": " << r.throughput_gbps
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

    RingAttnComm comm(rank, world_size);

    auto [correctness, rows] = runTest(
        comm,
        kBenchmarkN_per_dev,
        static_cast<int>(sizeof(kBenchmarkN_per_dev) / sizeof(kBenchmarkN_per_dev[0])),
        kCorrectN_per_dev,
        kNumWarmup,
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
    constexpr int WORLD_SIZE = attn_comm::globals::NUM_DEVICES;

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
