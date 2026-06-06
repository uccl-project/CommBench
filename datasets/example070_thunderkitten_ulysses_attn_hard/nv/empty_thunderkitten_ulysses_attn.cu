// HARD task — multi-GPU ulysses_attn from scratch (no ThunderKittens).
//
// Behavioral spec (mirrors the reference at `ref_thunderkitten_ulysses_attn.cu`):
// ThunderKittens BF16 multi-GPU Ulysses-Attention comm round (NVLink/TMA).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// Forks NUM_DEVICES (8) child processes; each pinned to one GPU brings up
// VMM + IPC + multicast through Broker, runs a correctness check
// and a per-iteration latency / bandwidth benchmark, and prints exactly
// one JSON object on rank 0.
//
// Note vs. the upstream `kernels/parallel/ulysses_attn/ulysses_attn.cu`:
//   * Upstream couples the four cross-rank all-to-all communications with
//     a FlashAttention-4 forward call (Python `flash_attn_fwd_raw`). The
//     attention compute is purely local — the only multi-GPU aspect of
//     Ulysses Attention is the all-to-all layout swap from sp→tp before
//     the FA call and the inverse tp→sp after it. This file is a
//     SIMPLIFIED COMM-ONLY port: the upstream `all_to_all` device kernel
//     is shipped verbatim and the four-call schedule is preserved
//     (3× sp→tp on Q/K/V with scatter=2 gather=1, then 1× tp→sp on O
//     with scatter=1 gather=2). The FlashAttention forward is omitted so
//     the example builds on B300 without a flash-attn-4 dependency, and
//     so the per-iteration cost reflects the comm-only TFLOP/s headroom
//     that Ulysses needs to hide behind the FA call.
//   * The four-kernel barrier-bracketed schedule, TMA tile sizes, and
//     thread/block geometry exactly match upstream — so the
//     comm-throughput numbers should be directly comparable.
//
// Code layout (per the dataset readme guidelines):
//   * Device-side kernels — `all_to_all::kernel<SCATTER, GATHER>` (TMA
//     load local tile, TMA store remote tile) and
//     `all_to_all_barrier::kernel` (multicast bracket barrier). Both are
//     verbatim from upstream.
//   * `ParallelTensor` — pure-C++ replacement for TKParallelTensor.
//   * `UlyssesAttn` — clean class. Owns the Broker and exposes
//     `all_to_all_sp_to_tp` / `all_to_all_tp_to_sp` / `ulysses_attn` /
//     `sync` / `zero` / `fill`.
//   * `runTest(...)` — correctness check (forward sp→tp on Q + reverse
//     tp→sp on O) + per-N benchmark of one full Ulysses comm round.
//   * `printJsonResult(...)` — emits exactly one JSON object on rank 0.
//   * `rank_main` / `main` — fork NUM_DEVICES children and wait.
//
// All shapes mirror the upstream `benchmark.py` Ulysses run:
//   Q_sp / K_sp / V_sp / O_sp : (1, N/W, H,   D)   bf16
//   Q_tp / K_tp / V_tp / O_tp : (1, N,   H/W, D)   bf16
// with H = D = 128 and W = NUM_DEVICES = 8.
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
// The reference at `ref_thunderkitten_ulysses_attn.cu` is the behavioral /
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
//   Device-side kernels (kernel bodies are intentionally identical to
//   the upstream ThunderKittens ulysses_attn.cu / all_to_all.cu — perf
//   must be unchanged).
// =====================================================================

namespace all_to_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;

    // TODO: define remaining kernel-launch shape parameters
    //       (NUM_THREADS, DYNAMIC_SHARED_MEMORY, etc.).
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;

    // TODO: define the kernel's globals (TMA descriptors,
    //       peer-pointer arrays, dev_idx, runtime params, plus
    //       any host-side helpers used by the launcher such as
    //       `dim3 grid()` / `int dynamic_shared_memory()`).
};

template <int SCATTER_AXIS, int GATHER_AXIS>
__device__ inline void kernel(const globals &G) {
    // TODO
}

} // namespace all_to_all

namespace all_to_all_barrier {

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

} // namespace all_to_all_barrier


// =====================================================================
//   Minimal launch helper (no cluster, no PDL, no torch stream).
//   Handles dynamic shared memory either as a constexpr Config field or
//   as a Globals::dynamic_shared_memory() member function.
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
//   Helper kernel for correctness fills.
// =====================================================================

__global__ void fill_bf16_kernel(__nv_bfloat16 *p, size_t n, __nv_bfloat16 v) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// Per-rank correctness tag: a permutation of the rank index (distinct for every
// rank, bf16-exact) rather than the trivially guessable `rank + 1`. The
// all-to-all check verifies each source rank's tag lands at the right
// destination, so a kernel cannot pass by writing a simple position-based
// constant — it must actually route the data.
static inline float ulysses_val(int r) {
    return 1.0f + static_cast<float>(((r * 5 + 3) & 7)) * 0.25f;
}


// =====================================================================
//   UlyssesAttn — clean communication class (no test/benchmark logic).
//
//   Owns the Broker and exposes:
//     zero / fill / all_to_all_sp_to_tp / all_to_all_tp_to_sp /
//     ulysses_attn / sync.
//
//   `ulysses_attn` runs the full comm round used by Ulysses Attention:
//     barrier -> a2a Q (sp→tp) -> barrier
//     barrier -> a2a K (sp→tp) -> barrier
//     barrier -> a2a V (sp→tp) -> barrier
//     [FlashAttention forward — OMITTED in this comm-only port]
//     barrier -> a2a O (tp→sp) -> barrier
// =====================================================================

class UlyssesAttn {
 public:
    UlyssesAttn(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    UlyssesAttn(const UlyssesAttn &) = delete;
    UlyssesAttn &operator=(const UlyssesAttn &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    Broker &broker() { return broker_; }

    void sync() { broker_.sync(); }

    void zero(ParallelTensor &t, size_t bytes) {
        cuda_check(cudaMemsetAsync(t.raw_ptrs[rank_], 0, bytes));
    }

    void fill(ParallelTensor &t, size_t numel, __nv_bfloat16 v) {
        const int threads = 256;
        const dim3 grid((numel + threads - 1) / threads);
        fill_bf16_kernel<<<grid, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(t.raw_ptrs[rank_]), numel, v);
    }

    // sp-layout (1, N/W, H,   D)  ->  tp-layout (1, N, H/W, D)  (scatter=2, gather=1)
    void all_to_all_sp_to_tp(ParallelTensor &tp, ParallelTensor &sp,
                             ParallelTensor &barrier,
                             int N_per_rank, int H, int D) {
        all_to_all_helper<2, 1>(/*output*/ tp, /*input*/ sp, barrier,
                                /*in_depth*/  N_per_rank, /*in_rows*/  H,
                                /*out_depth*/ N_per_rank * world_size_, /*out_rows*/ H / world_size_,
                                D);
    }

    // tp-layout (1, N, H/W, D)  ->  sp-layout (1, N/W, H,   D)  (scatter=1, gather=2)
    void all_to_all_tp_to_sp(ParallelTensor &sp, ParallelTensor &tp,
                             ParallelTensor &barrier,
                             int N_per_rank, int H, int D) {
        all_to_all_helper<1, 2>(/*output*/ sp, /*input*/ tp, barrier,
                                /*in_depth*/  N_per_rank * world_size_, /*in_rows*/  H / world_size_,
                                /*out_depth*/ N_per_rank, /*out_rows*/  H,
                                D);
    }

    // Full Ulysses Attention comm round (no FlashAttention compute).
    void ulysses_attn(ParallelTensor &Q_sp, ParallelTensor &Q_tp,
                      ParallelTensor &K_sp, ParallelTensor &K_tp,
                      ParallelTensor &V_sp, ParallelTensor &V_tp,
                      ParallelTensor &O_sp, ParallelTensor &O_tp,
                      ParallelTensor &barrier,
                      int N_per_rank, int H, int D) {
        all_to_all_sp_to_tp(Q_tp, Q_sp, barrier, N_per_rank, H, D);
        all_to_all_sp_to_tp(K_tp, K_sp, barrier, N_per_rank, H, D);
        all_to_all_sp_to_tp(V_tp, V_sp, barrier, N_per_rank, H, D);
        // [FlashAttention forward on (Q_tp, K_tp, V_tp) -> (O_tp, L) — OMITTED].
        all_to_all_tp_to_sp(O_sp, O_tp, barrier, N_per_rank, H, D);
    }

    static void cuda_check(cudaError_t err) {
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err));
    }

 private:
    template <int SCATTER, int GATHER>
    void all_to_all_helper(ParallelTensor &output, ParallelTensor &input,
                           ParallelTensor &barrier,
                           int in_depth, int in_rows,
                           int out_depth, int out_rows,
                           int D) {
        // TODO: implement using your kernel + Broker + ParallelTensor.
        //       (See `ref_thunderkitten_*.cu` for the upstream version.)
    }

    int rank_;
    int world_size_;
    Broker broker_;
};


// =====================================================================
//   Test harness (correctness + performance benchmark).
// =====================================================================

struct MetricRow {
    double data_size_mb;       // Q_sp shard size in MB (per rank)
    double throughput_gbps;    // per-rank comm bytes for one full round / time
    double latency_ms;         // wall time for one full Ulysses comm round
};

// Benchmark sweep. With H=D=128, W=8, each rank holds 8 buffers (Q_sp,
// Q_tp, K_sp, K_tp, V_sp, V_tp, O_sp, O_tp) of N*4096 bytes each. The
// largest entry is bounded by ~8 GB per rank (N=262144) so the example
// fits comfortably on a B300 (192 GB).
static constexpr int kBenchmarkNs[] = {8192, 16384, 32768, 65536, 131072, 262144};
static constexpr int kBenchmarkH    = 128;
static constexpr int kBenchmarkD    = 128;
static constexpr int kNumWarmupIters = 1;
static constexpr int kNumIters       = 5;

static constexpr int kCorrectnessN = 256;   // total N (must be world_size * something)
static constexpr int kCorrectnessH = 128;
static constexpr int kCorrectnessD = 128;


// Standalone test/benchmark function — handles correctness + timing.
//
//   * Correctness phase 1 — forward sp→tp:
//       Each rank fills Q_sp with `rank+1`; after one a2a(scatter=2,
//       gather=1), output[rank] should hold a stack of W contributions
//       along axis-1 (depth), with the contribution from source rank `s`
//       carrying the value `s+1`. We sample one cell from each source
//       rank's slot.
//
//   * Correctness phase 2 — reverse tp→sp:
//       Each rank fills O_tp with `rank+1`; after one a2a(scatter=1,
//       gather=2), output[rank] should hold a stack of W contributions
//       along axis-2 (rows), with the contribution from source rank `s`
//       carrying the value `s+1`. We sample one cell from each source
//       rank's row-block slot.
//
//   * Benchmark: for each N in `Ns`, run `warmup_iters` warmup iterations
//     + `iters` cudaEvent-timed iterations of the FULL Ulysses comm
//     round (3× a2a sp→tp on Q/K/V + 1× a2a tp→sp on O). Bandwidth is
//     reported as the per-rank bytes moved across the four collectives.
static std::pair<bool, std::vector<MetricRow>> runTest(
    UlyssesAttn &comm,
    const int *Ns,
    int num_sizes,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    const int W = comm.world_size();

    // ---- correctness phase 1: forward sp→tp on Q ----
    bool fwd_pass = false;
    {
        const int N = kCorrectnessN;
        const int H = kCorrectnessH;
        const int D = kCorrectnessD;
        const int N_per_rank = N / W;
        const int H_per_rank = H / W;

        const size_t sp_numel = static_cast<size_t>(1) * N_per_rank * H        * D;
        const size_t tp_numel = static_cast<size_t>(1) * N         * H_per_rank * D;
        const size_t sp_bytes = sp_numel * sizeof(__nv_bfloat16);
        const size_t tp_bytes = tp_numel * sizeof(__nv_bfloat16);

        ParallelTensor Q_sp(comm.broker(), sp_bytes, comm.rank(), W, false);
        ParallelTensor Q_tp(comm.broker(), tp_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), W, true);

        const __nv_bfloat16 fill_v =
            __float2bfloat16(ulysses_val(comm.rank()));
        comm.fill(Q_sp, sp_numel, fill_v);
        comm.zero(Q_tp, tp_bytes);
        comm.zero(barrier, sizeof(int));
        UlyssesAttn::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        comm.all_to_all_sp_to_tp(Q_tp, Q_sp, barrier, N_per_rank, H, D);
        UlyssesAttn::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        // Q_tp[rank] shape (1, N, H_per_rank, D). Source rank s lands at
        // depth in [N_per_rank*s, N_per_rank*(s+1)) carrying value (s+1).
        bool ok = true;
        __nv_bfloat16 host_v;
        const size_t H_p = static_cast<size_t>(H_per_rank);
        const size_t D_s = static_cast<size_t>(D);
        for (int src = 0; src < W; ++src) {
            const size_t depth_idx = static_cast<size_t>(N_per_rank) * src;
            const size_t lin_idx = ((static_cast<size_t>(0) * N + depth_idx)
                                    * H_p + 0) * D_s + 0;
            UlyssesAttn::cuda_check(cudaMemcpy(
                &host_v,
                reinterpret_cast<__nv_bfloat16 *>(Q_tp.raw_ptrs[comm.rank()]) + lin_idx,
                sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            const float got = __bfloat162float(host_v);
            const float want = ulysses_val(src);
            if (std::fabs(got - want) > 1e-2f) { ok = false; break; }
        }
        fwd_pass = ok;
        comm.sync();
    }

    // ---- correctness phase 2: reverse tp→sp on O ----
    bool rev_pass = false;
    {
        const int N = kCorrectnessN;
        const int H = kCorrectnessH;
        const int D = kCorrectnessD;
        const int N_per_rank = N / W;
        const int H_per_rank = H / W;

        const size_t sp_numel = static_cast<size_t>(1) * N_per_rank * H        * D;
        const size_t tp_numel = static_cast<size_t>(1) * N         * H_per_rank * D;
        const size_t sp_bytes = sp_numel * sizeof(__nv_bfloat16);
        const size_t tp_bytes = tp_numel * sizeof(__nv_bfloat16);

        ParallelTensor O_tp(comm.broker(), tp_bytes, comm.rank(), W, false);
        ParallelTensor O_sp(comm.broker(), sp_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), W, true);

        const __nv_bfloat16 fill_v =
            __float2bfloat16(ulysses_val(comm.rank()));
        comm.fill(O_tp, tp_numel, fill_v);
        comm.zero(O_sp, sp_bytes);
        comm.zero(barrier, sizeof(int));
        UlyssesAttn::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        comm.all_to_all_tp_to_sp(O_sp, O_tp, barrier, N_per_rank, H, D);
        UlyssesAttn::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        // O_sp[rank] shape (1, N_per_rank, H, D). Source rank s lands at
        // rows in [s*ROW, (s+1)*ROW) (one row block of ROW=16 lines)
        // carrying value (s+1). Sample at row = s*ROW, col = 0.
        bool ok = true;
        __nv_bfloat16 host_v;
        const size_t H_s = static_cast<size_t>(H);
        const size_t D_s = static_cast<size_t>(D);
        const int ROW = all_to_all::globals::ROW_BLOCK_SIZE;
        for (int src = 0; src < W; ++src) {
            const size_t row_idx = static_cast<size_t>(src) * ROW;
            const size_t lin_idx = ((static_cast<size_t>(0) * N_per_rank + 0)
                                    * H_s + row_idx) * D_s + 0;
            UlyssesAttn::cuda_check(cudaMemcpy(
                &host_v,
                reinterpret_cast<__nv_bfloat16 *>(O_sp.raw_ptrs[comm.rank()]) + lin_idx,
                sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            const float got = __bfloat162float(host_v);
            const float want = ulysses_val(src);
            if (std::fabs(got - want) > 1e-2f) { ok = false; break; }
        }
        rev_pass = ok;
        comm.sync();
    }

    const bool overall_pass = fwd_pass && rev_pass;

    // ---- benchmark: full 4-call Ulysses comm round ----
    const int H = kBenchmarkH;
    const int D = kBenchmarkD;
    const int H_per_rank = H / W;

    for (int i = 0; i < num_sizes; ++i) {
        const int N = Ns[i];
        const int N_per_rank = N / W;

        const size_t sp_numel = static_cast<size_t>(1) * N_per_rank * H        * D;
        const size_t tp_numel = static_cast<size_t>(1) * N         * H_per_rank * D;
        const size_t sp_bytes = sp_numel * sizeof(__nv_bfloat16);
        const size_t tp_bytes = tp_numel * sizeof(__nv_bfloat16);

        ParallelTensor Q_sp(comm.broker(), sp_bytes, comm.rank(), W, false);
        ParallelTensor Q_tp(comm.broker(), tp_bytes, comm.rank(), W, false);
        ParallelTensor K_sp(comm.broker(), sp_bytes, comm.rank(), W, false);
        ParallelTensor K_tp(comm.broker(), tp_bytes, comm.rank(), W, false);
        ParallelTensor V_sp(comm.broker(), sp_bytes, comm.rank(), W, false);
        ParallelTensor V_tp(comm.broker(), tp_bytes, comm.rank(), W, false);
        ParallelTensor O_sp(comm.broker(), sp_bytes, comm.rank(), W, false);
        ParallelTensor O_tp(comm.broker(), tp_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), W, true);

        // Non-zero, non-constant inputs so a kernel cannot fake fast timing by
        // detecting all-zero activations and skipping the all-to-all.
        comm.fill(Q_sp, sp_bytes / sizeof(__nv_bfloat16), __float2bfloat16(ulysses_val(comm.rank()))); comm.zero(Q_tp, tp_bytes);
        comm.fill(K_sp, sp_bytes / sizeof(__nv_bfloat16), __float2bfloat16(ulysses_val(comm.rank()))); comm.zero(K_tp, tp_bytes);
        comm.fill(V_sp, sp_bytes / sizeof(__nv_bfloat16), __float2bfloat16(ulysses_val(comm.rank()))); comm.zero(V_tp, tp_bytes);
        comm.zero(O_sp, sp_bytes); comm.zero(O_tp, tp_bytes);
        comm.zero(barrier, sizeof(int));
        UlyssesAttn::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            comm.ulysses_attn(Q_sp, Q_tp, K_sp, K_tp, V_sp, V_tp, O_sp, O_tp,
                              barrier, N_per_rank, H, D);
        UlyssesAttn::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        UlyssesAttn::cuda_check(cudaEventCreate(&start_evt));
        UlyssesAttn::cuda_check(cudaEventCreate(&stop_evt));
        UlyssesAttn::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            comm.ulysses_attn(Q_sp, Q_tp, K_sp, K_tp, V_sp, V_tp, O_sp, O_tp,
                              barrier, N_per_rank, H, D);
        UlyssesAttn::cuda_check(cudaEventRecord(stop_evt));
        UlyssesAttn::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        UlyssesAttn::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        UlyssesAttn::cuda_check(cudaEventDestroy(start_evt));
        UlyssesAttn::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms = total_ms / iters;
        // Per a2a, per-rank comm bytes (excluding self-chunk):
        //   chunk = (N/W) * (H/W) * D * sizeof(bf16)   [same for both directions]
        //   per_a2a = chunk * (W - 1)
        // The full Ulysses round runs 4 a2as (3 sp→tp + 1 tp→sp).
        const double chunk_bytes = static_cast<double>(N_per_rank)
                                 * static_cast<double>(H_per_rank)
                                 * static_cast<double>(D) * sizeof(__nv_bfloat16);
        const double per_rank_bytes = 4.0 * chunk_bytes * (W - 1);
        const double gbps = avg_ms > 0.0
            ? (per_rank_bytes * 1e-9) / (avg_ms * 1e-3)
            : 0.0;

        rows.push_back({
            static_cast<double>(sp_bytes) / 1e6,   // size of one Q_sp shard
            gbps,
            avg_ms,
        });

        comm.sync();
    }

    return {overall_pass, rows};
}


// =====================================================================
//   JSON output (rank 0 only, must be exactly one JSON object).
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

    UlyssesAttn comm(rank, world_size);

    auto [correctness, rows] = runTest(
        comm,
        kBenchmarkNs,
        static_cast<int>(sizeof(kBenchmarkNs) / sizeof(kBenchmarkNs[0])),
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
    constexpr int WORLD_SIZE = all_to_all::globals::NUM_DEVICES;

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
