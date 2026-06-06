// ThunderKittens BF16 multi-GPU MoE Dispatch + Grouped GEMM (NVLink/multicast).
// Empty template: complete the three device kernels marked `// TODO`.
//
// All host scaffolding (per-process fork-and-join, KittensBroker, VMM/IPC
// setup, deterministic top-k routing + `pull_dispatch_indices` generation,
// correctness check, perf benchmark, JSON output) is already wired up —
// the model only needs to fill the device-side `dispatch_ns::kernel`,
// `gemm_ns::kernel`, and `epilogue_ns::kernel` bodies.
//
// What you need to implement:
//   * `dispatch_ns::kernel(G)` — TMA-pull tokens from peer ranks and TMA-
//     store them into the local `post_tokens` buffer, then atomically
//     increment the per-row barrier counter. Each block handles
//     `TOKENS_PER_BLOCK` tokens (one thread per token, lanes 0..15):
//       - For each `token_idx = sm_idx * TOKENS_PER_BLOCK + lane`:
//           src_dev_idx   = G.pull_dispatch_indices[{token_idx, 0}]
//           src_token_idx = G.pull_dispatch_indices[{token_idx, 1}]
//           if src_dev_idx >= 0:
//               init_semaphore + tma::expect_bytes + tma::load_async from
//               G.pre_tokens[src_dev_idx] at row src_token_idx, then
//               wait + tma::store_async to G.post_tokens at row token_idx.
//           red.release.gpu.global.add.s32 [&G.barrier[G.dev_idx][{token_idx
//           / ROW_BLOCK}]], 1.
//   * `gemm_ns::kernel(g)` — simple grouped GEMM. Grid:
//        blockIdx.z = local_expert_id, blockIdx.y = row_block, blockIdx.x = col_block.
//     Each block produces a (BLOCK_M, BLOCK_N) tile of
//        outputs[expert_e][row_block, col_block] =
//            post_tokens[expert_e_rows] @ weights[expert_e].
//     Use `__shared__` tiles + a per-thread (RM, RN) fp32 accumulator and
//     iterate over K in BLOCK_K-wide chunks.
//   * `epilogue_ns::kernel(G)` — clear the dispatch barrier counters
//     (`G.barrier[G.dev_idx][{i}] = 0` for `i ∈ [0, G.num_blocks)`).
//
// Refer to `ref_thunderkitten_moe_dispatch_gemm.cu` (this directory) for
// the reference implementation. See its header for context on the
// simplifications vs. the upstream H100 kernel.
//
// Note vs. the upstream `moe_dispatch_gemm_h100.cu`:
//   * Upstream is Hopper-only (uses `warpgroup::mma_AB` / `wgmma`). The
//     wgmma instruction is not in the Blackwell ISA, and a real port to
//     `tcgen05.mma` is a substantial rewrite. This file is a SIMPLIFIED
//     port that focuses on getting the dispatch correct and providing a
//     functional (correctness-tested) end-to-end benchmark on B300.
//   * Upstream pre-computes `pull_dispatch_indices` in Python from a random
//     multinomial routing. To stay self-contained, this file uses a
//     deterministic top-k routing — token `i` goes to experts
//     `(i, i+1, ..., i+top_k-1) mod num_experts`. The result is a balanced
//     workload with `tokens_per_expert = B*S * top_k / num_experts`, which
//     makes `padded_tokens_per_expert` uniform across experts and the
//     `pull_dispatch_indices` table easy to compute on the host.
//   * The dispatch SM body is byte-for-byte the upstream `dispatch(...)`
//     (TMA pull from peer-rank `pre_tokens` + per-row barrier increment).
//     Both `tma::load_async`/`tma::store_async` and `red.release.gpu.global`
//     are Blackwell-compatible, so this is the only piece that retains
//     upstream-equivalent perf characteristics.
//   * The grouped GEMM is a STRAIGHTFORWARD scalar-MAC kernel — each block
//     computes a (BLOCK_M, BLOCK_N) output tile via shared-memory tile
//     loads + a per-thread RM×RN fp32 accumulator; one launch covers all
//     local experts via a 3D grid. This is intentionally portable rather
//     than peak: it gets the example running on B300 with correct numerics
//     but does NOT approach the upstream H100 wgmma TFLOP/s. A proper
//     Blackwell port would replace this kernel with a `tcgen05.mma`-based
//     producer/consumer pipeline; that is left as future work.
//
// Code layout:
//   * Device-side kernels — `dispatch_kernel` (per-token TMA pull + per-row
//     barrier counter), `grouped_gemm_kernel` (single-CTA tcgen05.mma per
//     tile), `epilogue_kernel` (zero out the dispatch barrier).
//   * `ParallelTensor` — pure-C++ replacement for kittens::py::TKParallelTensor.
//   * `DeviceBuffer` — RAII cudaMalloc wrapper for the non-shared tensors.
//   * `MoeDispatchGemm` — clean class. Owns the KittensBroker, exposes
//     `run` / `sync` and a `pre_tokens` ParallelTensor.
//   * `runTest(...)` — correctness check + TFLOP/s benchmark.
//   * `printJsonResult(...)` — emits exactly one JSON object on rank 0.
//   * `rank_main` / `main` — fork NUM_DEVICES children and wait.

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
//   Compile-time DeepSeek-V3 model dimensions (matching upstream).
// =====================================================================

namespace moe {

static constexpr int NUM_DEVICES         = 8;
static constexpr int H                   = 7168;   // hidden size
static constexpr int I                   = 2048;   // expert intermediate size
static constexpr int TOP_K               = 8;
static constexpr int NUM_EXPERTS         = 256;
static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / NUM_DEVICES;

// Dispatch tile.
using token_vec = sv_bf<H>;
static constexpr int TOKENS_PER_BLOCK = 16;

// Per-row barrier signaling chunk = ROW_BLOCK tokens (matches GEMM ROW_BLOCK).
static constexpr int ROW_BLOCK = 128;

// GEMM tile sizes (single-CTA tcgen05).
static constexpr int GEMM_M           = ROW_BLOCK;   // one block produces 128 rows of C
static constexpr int GEMM_N           = 256;         // COL_BLOCK
static constexpr int GEMM_K           = 64;          // RED_BLOCK (per reduction iter)
static constexpr int GEMM_EPI_DEPTH   = 8;           // split GEMM_N into 8 cols of N/8 each
static constexpr int GEMM_EPI_CHUNK_N = GEMM_N / GEMM_EPI_DEPTH;  // 32

using dispatch_pgl = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
// Separate post_tokens_gl per kernel — single TMA tile spec each, to keep
// the TMA descriptor unambiguous.
using post_tokens_dispatch_gl = gl<bf16, 1, 1, -1, H, token_vec>;
using post_tokens_gemm_gl     = gl<bf16, 1, 1, -1, H, st_bf<GEMM_M / 2, GEMM_K>>;
using weights_gl =
    gl<bf16, 1, NUM_EXPERTS_PER_DEV, H, I, st_bf<GEMM_K, GEMM_N>>;
using outputs_gl =
    gl<bf16, 1, 1, -1, I, st_bf<GEMM_M, GEMM_EPI_CHUNK_N>>;
using padded_per_expert_gl  = gl<int, 1, 1, 1, NUM_EXPERTS>;
using pull_indices_gl       = gl<int, 1, 1, -1, 2>;
using barrier_pgl           = barrier_t<NUM_DEVICES>;

} // namespace moe


// =====================================================================
//   Dispatch SM — TMA-pull tokens from peer ranks, write to local
//   `post_tokens`, then per-row counter increment so that the GEMM
//   block can spin-wait on its assigned ROW_BLOCK becoming complete.
//
//   This is byte-for-byte the upstream `dispatch(...)` body. The PTX
//   `red.release.gpu.global.add.s32` is a generic atomic-add; both this
//   and the surrounding TMA calls are Blackwell-compatible.
// =====================================================================

namespace dispatch_ns {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 384;
    static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;
};

struct globals {
    moe::dispatch_pgl pre_tokens;
    moe::post_tokens_dispatch_gl post_tokens;
    moe::pull_indices_gl pull_dispatch_indices;
    moe::barrier_pgl barrier;
    const int dev_idx;
    const int num_padded_local_tokens;
};

__device__ inline void kernel(const globals &G) {
    // TODO: TMA-pull dispatch — see file header for full spec.
    //   1. Allocate a `moe::token_vec` array of size `TOKENS_PER_BLOCK`
    //      in dynamic shared memory via `tma_swizzle_allocator`, plus a
    //      parallel `__shared__ semaphore token_arrived[TOKENS_PER_BLOCK]`.
    //   2. For lane in [0, TOKENS_PER_BLOCK):
    //        token_idx = blockIdx.x * TOKENS_PER_BLOCK + lane
    //        if (token_idx < G.num_padded_local_tokens):
    //            src_dev_idx   = G.pull_dispatch_indices[{token_idx, 0}]
    //            src_token_idx = G.pull_dispatch_indices[{token_idx, 1}]
    //            if (src_dev_idx >= 0 && src_token_idx >= 0):
    //                init_semaphore(token_arrived[lane], 0, 1)
    //                tma::expect_bytes(token_arrived[lane], sizeof(token_vec))
    //                tma::load_async(token[lane], G.pre_tokens[src_dev_idx],
    //                                {src_token_idx, 0}, token_arrived[lane])
    //                wait(token_arrived[lane], 0)
    //                tma::store_async(G.post_tokens, token[lane],
    //                                 {token_idx, 0})
    //                tma::store_async_wait()
    //            (always) red.release.gpu.global.add.s32
    //                     [&G.barrier[G.dev_idx][{token_idx / ROW_BLOCK}]], 1.
}

} // namespace dispatch_ns


// =====================================================================
//   Grouped GEMM kernel — Blackwell single-CTA tcgen05.mma.
//
//   Grid:
//     blockIdx.z = local_expert_id  ∈ [0, NUM_EXPERTS_PER_DEV)
//     blockIdx.y = row_block_idx    ∈ [0, padded_M_per_expert / ROW_BLOCK)
//     blockIdx.x = col_block_idx    ∈ [0, I / COL_BLOCK)
//
//   Each block:
//     1. Spin-waits on `barrier[expert_row_offset / ROW_BLOCK + row_block_idx]
//        == ROW_BLOCK` so that the dispatch SMs have finished filling the
//        row tile this block consumes.
//     2. Pipeline-loads A and B tiles, accumulates K reductions into a
//        `tt<float, ROW_BLOCK, COL_BLOCK>` tensor-memory accumulator with
//        `tcgen05.mma_ABt`.
//     3. After the inner loop, commits + waits, reads tensor memory back
//        into a register tile (with fp32→bf16 conversion), and stores the
//        result to `outputs` via `tma::store_async`.
// =====================================================================

namespace gemm_ns {

// Block dims for the simple GEMM:
//   each thread block computes a (BLOCK_M, BLOCK_N) output tile, with each
//   thread covering (RM, RN) elements via a scalar accumulation loop.
//
// We deliberately keep the GEMM kernel architecturally trivial (no TMA,
// no tcgen05) so the example builds and runs reliably on any Blackwell
// SKU. Performance will be substantially below the upstream H100 wgmma
// kernel — see the file header for context.
static constexpr int BLOCK_M = 32;
static constexpr int BLOCK_N = 64;
static constexpr int BLOCK_K = 32;
static constexpr int TM      = 8;
static constexpr int TN      = 16;
static constexpr int RM      = BLOCK_M / TM;  // 4
static constexpr int RN      = BLOCK_N / TN;  // 4

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = TM * TN;          // 128
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    using A_gl = gl<bf16, 1, 1, -1, moe::H>;
    using B_gl = gl<bf16, 1, moe::NUM_EXPERTS_PER_DEV, moe::H, moe::I>;
    using C_gl = gl<bf16, 1, 1, -1, moe::I>;
    using barrier_pgl = moe::barrier_pgl;

    A_gl post_tokens;
    B_gl weights;
    C_gl outputs;
    barrier_pgl barrier;

    const int dev_idx;
    const int padded_tokens_per_expert;
};

__device__ inline void kernel(const globals &g) {
    // TODO: simple grouped GEMM — see file header for full spec.
    //
    //   1. Compute scheduling indices:
    //        local_expert_id = blockIdx.z
    //        row_block_idx   = blockIdx.y
    //        col_block_idx   = blockIdx.x
    //        tid_y = threadIdx.x / TN, tid_x = threadIdx.x % TN
    //        row_base = local_expert_id * g.padded_tokens_per_expert
    //                 + row_block_idx * BLOCK_M
    //        col_base = col_block_idx * BLOCK_N
    //   2. Allocate a per-thread fp32 register accumulator
    //      `float acc[RM][RN] = 0`.
    //   3. Allocate `__shared__ __nv_bfloat16 A_smem[BLOCK_M][BLOCK_K]`
    //      and `__shared__ __nv_bfloat16 B_smem[BLOCK_K][BLOCK_N]`.
    //   4. For kt in [0, moe::H / BLOCK_K):
    //        Cooperative load A tile from
    //          g.post_tokens[row_base..row_base+BLOCK_M, kt*BLOCK_K..+BLOCK_K]
    //        and B tile from
    //          g.weights[local_expert_id, kt*BLOCK_K..+BLOCK_K, col_base..+BLOCK_N]
    //        into shared memory; __syncthreads().
    //        For kk in [0, BLOCK_K), accumulate
    //            acc[i][j] += A_smem[tid_y*RM+i][kk] * B_smem[kk][tid_x*RN+j]
    //        for each (i, j) ∈ [0, RM) × [0, RN); __syncthreads().
    //   5. Write `acc[i][j]` (fp32 → bf16) to
    //         g.outputs[row_base + tid_y*RM + i, col_base + tid_x*RN + j].
    (void)g;
}

} // namespace gemm_ns


// =====================================================================
//   Epilogue kernel — zero out the dispatch barrier counters.
// =====================================================================

namespace epilogue_ns {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    moe::barrier_pgl barrier;
    const int dev_idx;
    const int num_blocks;
};

__device__ inline void kernel(const globals &G) {
    // TODO: zero out the dispatch barrier counters so the next iteration
    // starts fresh. Strided loop:
    //   for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    //        i < G.num_blocks;
    //        i += gridDim.x * blockDim.x)
    //       G.barrier[G.dev_idx][{i}] = 0;
}

} // namespace epilogue_ns


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
//   DeviceBuffer (cudaMalloc RAII).
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

// Deterministic, non-constant, non-zero test pattern (values in {0.5,1.0,1.5,2.0},
// all bf16-exact). Lets a correctness reference be computed in closed form while
// inputs stay unpredictable — defeats kernels that detect trivial (all-zero /
// all-equal) inputs or hardcode the expected output.
__host__ __device__ inline float tk_pattern(unsigned long long idx, unsigned long long seed) {
    unsigned long long x = (idx + 1ULL) * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x ^= x >> 31;
    return 0.5f * static_cast<float>(1u + static_cast<unsigned>(x & 3ULL));
}

// Fill p[idx] = scale * tk_pattern(base + (idx % row_len), seed). With row_len=H
// the value depends only on the hidden index (token-independent); with
// row_len=H*I it is expert-independent. `scale` keeps the GEMM output small so
// bf16 accumulation stays accurate.
__global__ void fill_moe_row_kernel(__nv_bfloat16 *p, size_t n, unsigned long long row_len,
                                    unsigned long long base, unsigned long long seed, float scale) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2bfloat16(scale * tk_pattern(base + (static_cast<unsigned long long>(i) % row_len), seed));
}


// =====================================================================
//   Deterministic top-k routing pull-index generator (host-side).
//
//   Each global token i ∈ [0, B*S) routes to experts
//       (i, i+1, ..., i+TOP_K-1) mod NUM_EXPERTS.
//   With NUM_EXPERTS divisible by TOP_K and B*S divisible by NUM_EXPERTS,
//   every expert receives exactly `B*S * TOP_K / NUM_EXPERTS` tokens.
//   pull_dispatch_indices[k, *] for the k-th token of local_expert e
//   (e ∈ [rank * NEPD, (rank+1) * NEPD)) refers to global token index
//   `e + j * NUM_EXPERTS` (for j in [0, ...)) — a strided pattern.
// =====================================================================

static void compute_dispatch_indices(int *host_indices,
                                     int rank, int world_size,
                                     int B, int S,
                                     int padded_tokens_per_expert) {
    const int NUM_EXPERTS = moe::NUM_EXPERTS;
    const int TOP_K       = moe::TOP_K;
    const int NEPD        = moe::NUM_EXPERTS_PER_DEV;
    const int total_tokens = B * S;
    const int num_init_tokens_per_dev = total_tokens / world_size;

    const int expert_start = rank * NEPD;
    const int expert_end   = (rank + 1) * NEPD;

    int write_idx = 0;
    for (int local_e = 0; local_e < NEPD; ++local_e) {
        const int expert = expert_start + local_e;

        int count_for_this_expert = 0;
        // Walk every global token; a token belongs to this expert iff
        // expert ∈ {(i+0)..(i+TOP_K-1)} mod NUM_EXPERTS, i.e.
        // i mod NUM_EXPERTS ∈ {expert, expert-1, ..., expert-TOP_K+1} mod NUM_EXPERTS.
        for (int i = 0; i < total_tokens; ++i) {
            int delta = (expert - (i % NUM_EXPERTS) + NUM_EXPERTS) % NUM_EXPERTS;
            if (delta < TOP_K) {
                int src_dev_idx   = i / num_init_tokens_per_dev;
                int src_token_idx = i % num_init_tokens_per_dev;
                host_indices[2 * write_idx + 0] = src_dev_idx;
                host_indices[2 * write_idx + 1] = src_token_idx;
                ++write_idx;
                ++count_for_this_expert;
            }
        }

        // Sentinel-fill the padding slots (-1, -1).
        const int slot_end = (local_e + 1) * padded_tokens_per_expert;
        while (write_idx < slot_end) {
            host_indices[2 * write_idx + 0] = -1;
            host_indices[2 * write_idx + 1] = -1;
            ++write_idx;
        }
        (void)expert_end;
        (void)count_for_this_expert;
    }
}


// =====================================================================
//   MoeDispatchGemm — clean communication class.
// =====================================================================

class MoeDispatchGemm {
 public:
    MoeDispatchGemm(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    MoeDispatchGemm(const MoeDispatchGemm &) = delete;
    MoeDispatchGemm &operator=(const MoeDispatchGemm &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    KittensBroker &broker() { return broker_; }
    void sync() { broker_.sync(); }

    // Run the dispatch kernel + grouped GEMM kernel + barrier-clear.
    //
    //   * `pre_tokens`  : (num_init_tokens_per_dev, H) bf16 IPC tensor
    //                     (per-rank shard of the input — peers TMA-pull
    //                     from this via the multicast/IPC handles).
    //   * `post_tokens` : (num_padded_local_tokens, H) bf16 plain device buf,
    //                     filled by the dispatch kernel.
    //   * `weights`     : (NUM_EXPERTS_PER_DEV, H, I) bf16 plain device buf.
    //   * `outputs`     : (num_padded_local_tokens, I) bf16 plain device buf.
    //   * `pull_idx`    : (num_padded_local_tokens, 2) int32 plain device buf.
    //   * `barrier`     : (num_row_blocks,) int IPC tensor
    //                     — per-row-tile counter incremented by dispatch
    //                     SMs and spin-waited on by GEMM SMs.
    //   * `padded_tokens_per_expert` : uniform constant.
    void run(ParallelTensor &pre_tokens,
             DeviceBuffer &post_tokens, DeviceBuffer &weights,
             DeviceBuffer &outputs, DeviceBuffer &pull_idx,
             ParallelTensor &barrier,
             int num_padded_local_tokens, int padded_tokens_per_expert) {
        // ---- dispatch kernel ----
        {
            namespace ns = dispatch_ns;
            namespace mm = moe;

            auto pre_pgl = make_pgl<mm::dispatch_pgl>(
                reinterpret_cast<uint64_t *>(pre_tokens.raw_ptrs),
                1, 1, num_init_tokens_per_dev_, mm::H);
            auto post_gl = make_gl<mm::post_tokens_dispatch_gl>(
                reinterpret_cast<uint64_t>(post_tokens.ptr()),
                1, 1, num_padded_local_tokens, mm::H);
            auto pull_gl = make_gl<mm::pull_indices_gl>(
                reinterpret_cast<uint64_t>(pull_idx.ptr()),
                1, 1, num_padded_local_tokens, 2);

            using br_t = mm::barrier_pgl;
            br_t br_pgl = make_pgl<br_t>(
                reinterpret_cast<uint64_t>(barrier.mc_ptr),
                reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
                1, 1, 1, num_padded_local_tokens / mm::ROW_BLOCK);

            ns::globals g{
                .pre_tokens = pre_pgl,
                .post_tokens = post_gl,
                .pull_dispatch_indices = pull_gl,
                .barrier = br_pgl,
                .dev_idx = rank_,
                .num_padded_local_tokens = num_padded_local_tokens,
            };

            const int dispatch_blocks =
                (num_padded_local_tokens + mm::TOKENS_PER_BLOCK - 1) / mm::TOKENS_PER_BLOCK;
            launch_kernel<ns::config, ns::globals, ns::kernel>(
                dim3(dispatch_blocks, 1, 1), g);
        }

        // ---- grouped GEMM kernel ----
        {
            namespace ns = gemm_ns;
            namespace mm = moe;

            auto post_gl = make_gl<ns::globals::A_gl>(
                reinterpret_cast<uint64_t>(post_tokens.ptr()),
                1, 1, num_padded_local_tokens, mm::H);
            auto wts_gl = make_gl<ns::globals::B_gl>(
                reinterpret_cast<uint64_t>(weights.ptr()),
                1, mm::NUM_EXPERTS_PER_DEV, mm::H, mm::I);
            auto out_gl = make_gl<ns::globals::C_gl>(
                reinterpret_cast<uint64_t>(outputs.ptr()),
                1, 1, num_padded_local_tokens, mm::I);

            using br_t = mm::barrier_pgl;
            br_t br_pgl = make_pgl<br_t>(
                reinterpret_cast<uint64_t>(barrier.mc_ptr),
                reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
                1, 1, 1, num_padded_local_tokens / mm::ROW_BLOCK);

            ns::globals g{
                .post_tokens = post_gl,
                .weights = wts_gl,
                .outputs = out_gl,
                .barrier = br_pgl,
                .dev_idx = rank_,
                .padded_tokens_per_expert = padded_tokens_per_expert,
            };

            const int row_blocks = padded_tokens_per_expert / ns::BLOCK_M;
            const int col_blocks = mm::I / ns::BLOCK_N;
            dim3 grid(col_blocks, row_blocks, mm::NUM_EXPERTS_PER_DEV);
            launch_kernel<ns::config, ns::globals, ns::kernel>(grid, g);
        }

        // ---- epilogue: clear barrier ----
        {
            namespace ns = epilogue_ns;
            namespace mm = moe;

            using br_t = mm::barrier_pgl;
            br_t br_pgl = make_pgl<br_t>(
                reinterpret_cast<uint64_t>(barrier.mc_ptr),
                reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
                1, 1, 1, num_padded_local_tokens / mm::ROW_BLOCK);

            ns::globals g{
                .barrier = br_pgl,
                .dev_idx = rank_,
                .num_blocks = num_padded_local_tokens / mm::ROW_BLOCK,
            };
            const int blocks = (g.num_blocks + ns::config::NUM_THREADS - 1)
                             / ns::config::NUM_THREADS;
            launch_kernel<ns::config, ns::globals, ns::kernel>(
                dim3(blocks, 1, 1), g);
        }
    }

    void set_num_init_tokens_per_dev(int v) { num_init_tokens_per_dev_ = v; }

    static void cuda_check(cudaError_t err) {
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err));
    }

 private:
    int rank_;
    int world_size_;
    int num_init_tokens_per_dev_ = 0;
    KittensBroker broker_;
};


// =====================================================================
//   Test harness (correctness + performance benchmark).
// =====================================================================

struct MetricRow {
    double data_size_mb;       // size of `outputs` per rank, in MB
    double throughput_tflops;  // 2 * (B*S*top_k) * H * I / world_size / time
    double latency_ms;
};

// 6 distinct seq_lens (matches the `>=6 sizes` requirement). Smaller values
// than upstream so the example finishes well under a minute end-to-end on
// B300 — every iteration touches H*I weights = 7168*2048 = ~14M bf16 = 28MB.
static constexpr int kBenchmarkSeqLens[] = {1024, 2048, 4096, 8192, 16384, 32768};
static constexpr int kBatchSize       = 1;
static constexpr int kNumWarmupIters  = 1;
static constexpr int kNumIters        = 5;
static constexpr int kCorrectnessSeq  = 1024;

// Correctness: pre_tokens filled with a per-rank constant `(rank+1) * 1e-3`,
// weights filled with `1e-2`. After dispatch+GEMM, every output element
// satisfies
//     C[token, j] = sum_k pre_token[k] * weight[k, j]
//                 = H * src_pre_token_value * 1e-2
//                 = H * (src_dev_idx + 1) * 1e-3 * 1e-2
//                 = H * (src_dev_idx + 1) * 1e-5
// where `src_dev_idx` is the rank that produced this token (recorded in
// the pull-index table). We sample a handful of rows and look up the
// expected `src_dev_idx` to check.
static std::pair<bool, std::vector<MetricRow>> runTest(
    MoeDispatchGemm &comm,
    const int *seq_lens,
    int num_seq_lens,
    int correctness_seq,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_seq_lens);

    const int W = comm.world_size();
    const int B = kBatchSize;

    auto fill = [&](void *p, size_t numel, __nv_bfloat16 v) {
        const int threads = 256;
        const dim3 grid((numel + threads - 1) / threads);
        fill_bf16_kernel<<<grid, threads>>>(reinterpret_cast<__nv_bfloat16 *>(p), numel, v);
    };

    auto setup_dispatch_buffers =
        [&](int B, int S, int &num_init_tokens_per_dev,
            int &num_padded_local_tokens, int &padded_tokens_per_expert,
            std::vector<int> &host_pull_indices) {
            const int total_tokens = B * S;
            num_init_tokens_per_dev = total_tokens / W;
            const int tokens_per_expert =
                (total_tokens * moe::TOP_K) / moe::NUM_EXPERTS;
            padded_tokens_per_expert =
                ((tokens_per_expert + moe::ROW_BLOCK - 1) / moe::ROW_BLOCK) * moe::ROW_BLOCK;
            num_padded_local_tokens =
                padded_tokens_per_expert * moe::NUM_EXPERTS_PER_DEV;
            host_pull_indices.assign(num_padded_local_tokens * 2, -1);
            compute_dispatch_indices(host_pull_indices.data(),
                                     comm.rank(), W, B, S,
                                     padded_tokens_per_expert);
        };

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int S = correctness_seq;
        int num_init_tokens_per_dev, num_padded_local_tokens, padded_tokens_per_expert;
        std::vector<int> host_pull_indices;
        setup_dispatch_buffers(B, S, num_init_tokens_per_dev,
                               num_padded_local_tokens,
                               padded_tokens_per_expert, host_pull_indices);
        comm.set_num_init_tokens_per_dev(num_init_tokens_per_dev);

        const size_t pre_bytes  = static_cast<size_t>(num_init_tokens_per_dev)
                                * moe::H * sizeof(__nv_bfloat16);
        const size_t post_bytes = static_cast<size_t>(num_padded_local_tokens)
                                * moe::H * sizeof(__nv_bfloat16);
        const size_t wts_bytes  = static_cast<size_t>(moe::NUM_EXPERTS_PER_DEV)
                                * moe::H * moe::I * sizeof(__nv_bfloat16);
        const size_t out_bytes  = static_cast<size_t>(num_padded_local_tokens)
                                * moe::I * sizeof(__nv_bfloat16);
        const size_t pull_bytes = host_pull_indices.size() * sizeof(int);
        const size_t bar_bytes  = std::max<size_t>(2u * 1024u * 1024u * sizeof(int),
                                                   (num_padded_local_tokens / moe::ROW_BLOCK) * sizeof(int));

        ParallelTensor pre_tokens(comm.broker(), pre_bytes, comm.rank(), W, false);
        DeviceBuffer post_tokens(post_bytes);
        DeviceBuffer weights(wts_bytes);
        DeviceBuffer outputs(out_bytes);
        DeviceBuffer pull_idx(pull_bytes);
        ParallelTensor barrier(comm.broker(), bar_bytes, comm.rank(), W, true);

        // Seeded, non-constant, non-zero inputs (defeats hardcode / all-equal /
        // zero-skip exploits). pre_tokens[token,h] = kPreScale*tk_pattern(rank*H+h)
        // is token-independent, so a dispatched token's value depends only on its
        // source device. weights[expert,h,i] = kWtScale*tk_pattern(h*I+i) is
        // expert-independent. After dispatch+GEMM,
        //   output[t,i] = sum_h pre[src_dev(t),h] * w[h,i].
        constexpr float kPreScale = 1.0f / 2048.0f;
        constexpr float kWtScale  = 1.0f / 128.0f;
        const size_t pre_numel = pre_bytes / sizeof(__nv_bfloat16);
        const size_t wt_numel  = wts_bytes / sizeof(__nv_bfloat16);
        fill_moe_row_kernel<<<dim3((pre_numel + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(pre_tokens.raw_ptrs[comm.rank()]), pre_numel,
            static_cast<unsigned long long>(moe::H),
            static_cast<unsigned long long>(comm.rank()) * moe::H, 0x1111ULL, kPreScale);
        fill_moe_row_kernel<<<dim3((wt_numel + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(weights.ptr()), wt_numel,
            static_cast<unsigned long long>(moe::H) * moe::I, 0ULL, 0x2222ULL, kWtScale);
        MoeDispatchGemm::cuda_check(cudaMemsetAsync(post_tokens.ptr(), 0, post_bytes));
        MoeDispatchGemm::cuda_check(cudaMemsetAsync(outputs.ptr(), 0, out_bytes));
        MoeDispatchGemm::cuda_check(cudaMemcpy(
            pull_idx.ptr(), host_pull_indices.data(), pull_bytes,
            cudaMemcpyHostToDevice));
        MoeDispatchGemm::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, bar_bytes));
        MoeDispatchGemm::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        comm.run(pre_tokens, post_tokens, weights, outputs, pull_idx, barrier,
                 num_padded_local_tokens, padded_tokens_per_expert);
        MoeDispatchGemm::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        // Sample three valid (non-padded) tokens; check
        //   C[token, 0] ≈ H * (host_pull_indices[token, 0] + 1) * 1e-5.
        const size_t sample_token_idxs[3] = {
            0,
            static_cast<size_t>(num_padded_local_tokens / 2),
            static_cast<size_t>(num_padded_local_tokens - 1),
        };
        overall_pass = true;
        for (int s = 0; s < 3; ++s) {
            size_t t = sample_token_idxs[s];
            int src_dev = host_pull_indices[2 * t + 0];
            if (src_dev < 0) {
                // Slot is padded; skip — no correctness claim.
                continue;
            }
            __nv_bfloat16 c_val_bf{};
            MoeDispatchGemm::cuda_check(cudaMemcpy(
                &c_val_bf,
                reinterpret_cast<__nv_bfloat16 *>(outputs.ptr()) + t * moe::I,
                sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));
            float c_val = __bfloat162float(c_val_bf);
            // Reference for output column 0: sum_h pre[src_dev,h] * w[h,0].
            double acc = 0.0;
            for (int h = 0; h < moe::H; ++h)
                acc += static_cast<double>(kPreScale * tk_pattern(
                           static_cast<unsigned long long>(src_dev) * moe::H + h, 0x1111ULL))
                     * static_cast<double>(kWtScale * tk_pattern(
                           static_cast<unsigned long long>(h) * moe::I + 0, 0x2222ULL));
            float expected = static_cast<float>(acc);
            // bf16 rounding + bf16 GEMM accumulation noise ≈ a few percent
            // for these magnitudes; allow a generous tolerance.
            if (std::fabs(c_val - expected) > expected * 0.10f + 1e-3f) {
                overall_pass = false;
                break;
            }
        }
        comm.sync();
    }

    // ---- benchmark ----
    for (int i = 0; i < num_seq_lens; ++i) {
        const int S = seq_lens[i];
        int num_init_tokens_per_dev, num_padded_local_tokens, padded_tokens_per_expert;
        std::vector<int> host_pull_indices;
        setup_dispatch_buffers(B, S, num_init_tokens_per_dev,
                               num_padded_local_tokens,
                               padded_tokens_per_expert, host_pull_indices);
        comm.set_num_init_tokens_per_dev(num_init_tokens_per_dev);

        const size_t pre_bytes  = static_cast<size_t>(num_init_tokens_per_dev)
                                * moe::H * sizeof(__nv_bfloat16);
        const size_t post_bytes = static_cast<size_t>(num_padded_local_tokens)
                                * moe::H * sizeof(__nv_bfloat16);
        const size_t wts_bytes  = static_cast<size_t>(moe::NUM_EXPERTS_PER_DEV)
                                * moe::H * moe::I * sizeof(__nv_bfloat16);
        const size_t out_bytes  = static_cast<size_t>(num_padded_local_tokens)
                                * moe::I * sizeof(__nv_bfloat16);
        const size_t pull_bytes = host_pull_indices.size() * sizeof(int);
        const size_t bar_bytes  = std::max<size_t>(2u * 1024u * 1024u * sizeof(int),
                                                   (num_padded_local_tokens / moe::ROW_BLOCK) * sizeof(int));

        ParallelTensor pre_tokens(comm.broker(), pre_bytes, comm.rank(), W, false);
        DeviceBuffer post_tokens(post_bytes);
        DeviceBuffer weights(wts_bytes);
        DeviceBuffer outputs(out_bytes);
        DeviceBuffer pull_idx(pull_bytes);
        ParallelTensor barrier(comm.broker(), bar_bytes, comm.rank(), W, true);

        // Non-zero, non-constant inputs so a kernel cannot fake fast timing by
        // detecting all-zero data and skipping the dispatch + matmul.
        fill_moe_row_kernel<<<dim3((pre_bytes / sizeof(__nv_bfloat16) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(pre_tokens.raw_ptrs[comm.rank()]),
            pre_bytes / sizeof(__nv_bfloat16), static_cast<unsigned long long>(moe::H),
            static_cast<unsigned long long>(comm.rank()) * moe::H, 0x1111ULL, 1.0f / 2048.0f);
        fill_moe_row_kernel<<<dim3((wts_bytes / sizeof(__nv_bfloat16) + 255) / 256), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(weights.ptr()), wts_bytes / sizeof(__nv_bfloat16),
            static_cast<unsigned long long>(moe::H) * moe::I, 0ULL, 0x2222ULL, 1.0f / 128.0f);
        MoeDispatchGemm::cuda_check(cudaMemsetAsync(post_tokens.ptr(), 0, post_bytes));
        MoeDispatchGemm::cuda_check(cudaMemsetAsync(outputs.ptr(), 0, out_bytes));
        MoeDispatchGemm::cuda_check(cudaMemcpy(
            pull_idx.ptr(), host_pull_indices.data(), pull_bytes,
            cudaMemcpyHostToDevice));
        MoeDispatchGemm::cuda_check(cudaMemsetAsync(barrier.raw_ptrs[comm.rank()], 0, bar_bytes));
        MoeDispatchGemm::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            comm.run(pre_tokens, post_tokens, weights, outputs, pull_idx, barrier,
                     num_padded_local_tokens, padded_tokens_per_expert);
        MoeDispatchGemm::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        MoeDispatchGemm::cuda_check(cudaEventCreate(&start_evt));
        MoeDispatchGemm::cuda_check(cudaEventCreate(&stop_evt));
        MoeDispatchGemm::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            comm.run(pre_tokens, post_tokens, weights, outputs, pull_idx, barrier,
                     num_padded_local_tokens, padded_tokens_per_expert);
        MoeDispatchGemm::cuda_check(cudaEventRecord(stop_evt));
        MoeDispatchGemm::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        MoeDispatchGemm::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        MoeDispatchGemm::cuda_check(cudaEventDestroy(start_evt));
        MoeDispatchGemm::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms = total_ms / iters;
        // Per-rank flops = 2 * (B * S * top_k) * H * I / world_size
        const double total_flops = 2.0 * static_cast<double>(B) * S * moe::TOP_K
                                 * moe::H * moe::I / W;
        const double tflops = avg_ms > 0.0
            ? total_flops * 1e-12 / (avg_ms * 1e-3)
            : 0.0;

        rows.push_back({
            static_cast<double>(out_bytes) / 1e6,
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

    MoeDispatchGemm comm(rank, world_size);

    auto [correctness, rows] = runTest(
        comm,
        kBenchmarkSeqLens,
        static_cast<int>(sizeof(kBenchmarkSeqLens) / sizeof(kBenchmarkSeqLens[0])),
        kCorrectnessSeq,
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
    constexpr int WORLD_SIZE = moe::NUM_DEVICES;

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
