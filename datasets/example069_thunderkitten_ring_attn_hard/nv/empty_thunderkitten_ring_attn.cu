// HARD task — multi-GPU ring_attn from scratch (no ThunderKittens).
//
// Behavioral spec (mirrors the reference at `ref_thunderkitten_ring_attn.cu`):
// ThunderKittens BF16 multi-GPU Ring-Attention K/V rotation (NVLink/IPC).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// Forks NUM_DEVICES (8) child processes; each pinned to one GPU brings up
// VMM + IPC + multicast through Broker, runs a correctness check
// and a TFLOP/s benchmark, prints exactly one JSON object on rank 0.
//
// Note vs. the upstream `ring_attn_h100.cu`:
//   * Upstream is Hopper-only (uses `warpgroup::mma_AB` / `wgmma` for the
//     two attention MMAs). The Blackwell ISA does not have wgmma, and a
//     real port to `tcgen05.mma` is a substantial rewrite (online softmax
//     + 2 MMA chains driven by tensor-memory accumulators). This file is
//     a SIMPLIFIED COMM-ONLY port that ships the upstream `attn_comm`
//     sub-kernel verbatim — i.e. the cross-rank K/V ring rotation that is
//     the unique multi-GPU aspect of ring attention. The attention
//     compute (`attn_partial` and `attn_reduction` upstream) is omitted.
//   * `attn_comm` only uses TMA loads / TMA stores / shared-memory
//     mbarriers, all of which are Blackwell-compatible — perf for the
//     rotation matches upstream.
//   * The benchmark drives the same 8-stage ring schedule (one stage per
//     peer in the world): each stage TMA-rotates K and V tiles from rank
//     `r` to rank `(r+1) mod NUM_DEVICES`; double-buffering between K0/K1
//     (and V0/V1) hides the in-flight buffer from the next stage.
//
// Code layout:
//   * Device-side kernels — `attn_comm::kernel` (TMA-rotate K and V tiles
//     to the next-ring rank), `barrier::kernel` (`barrier_all`).
//   * `ParallelTensor` — pure-C++ replacement for TKParallelTensor.
//   * `RingAttnComm` — clean class. Owns the Broker and exposes
//     `run_one_stage` / `sync`.
//   * `runTest(...)` — correctness check + bandwidth benchmark.
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
// The reference at `ref_thunderkitten_ring_attn.cu` is the behavioral /
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
//   ring_attn_h100.cu's `attn_comm` — perf must be unchanged)
// =====================================================================

namespace attn_comm {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - STATIC_SHARED_MEMORY;
    static constexpr int NUM_THREADS = 4 * WARPGROUP_WARPS * WARP_THREADS;

    // TODO: define remaining kernel-launch shape parameters
    //       (NUM_THREADS, DYNAMIC_SHARED_MEMORY, etc.).
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int D = 128;
    static constexpr int KV_BLOCK = 128;

    // TODO: define the kernel's globals (TMA descriptors,
    //       peer-pointer arrays, dev_idx, runtime params, plus
    //       any host-side helpers used by the launcher such as
    //       `dim3 grid()` / `int dynamic_shared_memory()`).
};

__device__ inline void kernel(const globals &G) {
    // TODO
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
    Broker &broker() { return broker_; }
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
