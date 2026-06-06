// HARD task — multi-GPU alltoall from scratch (no ThunderKittens).
//
// Behavioral spec (mirrors the reference at `ref_thunderkitten_alltoall.cu`):
// ThunderKittens BF16 multi-GPU AllToAll (NVLink/TMA, scatter=2 gather=1).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// `int main()` forks NUM_DEVICES (8) child processes, each pinned to one GPU.
// Each child uses the ThunderKittens VMM/IPC helpers (CUDA driver API)
// together with a POSIX-socket-backed Broker to allocate shareable
// device memory and exchange handles for peer access. The benchmark performs
// the same TMA-based all_to_all kernel as upstream
// `kernels/parallel/all_to_all/all_to_all.cu` (one tile per block, 1 thread
// per block, multicast barrier in/out), so the per-iteration latency /
// bandwidth on a B200/B300 host should match the upstream torchrun version.
//
// Code layout (per the dataset readme guidelines):
//   * Device-side kernels — `all_to_all::kernel<SCATTER, GATHER>` (TMA load
//     local tile / TMA store remote tile) and `all_to_all_barrier::kernel`
//     (multicast bracket barrier).
//   * `ParallelTensor` — pure-C++ replacement for TKParallelTensor;
//     allocates VMM-backed shareable physical memory, exchanges POSIX fds,
//     and (optionally) lays a multicast handle on top.
//   * `AllToAll` — clean communication class. Owns the Broker and
//     exposes only the core ops (zero / fill / all_to_all / sync). No test
//     or benchmark logic embedded.
//   * `runTest(...)` — dedicated standalone function. Runs the correctness
//     check at a small (N,H,D) and the latency/throughput benchmark across
//     kBenchmarkNs (1 warmup + kNumIters timed iterations per size,
//     cudaEvent timed).
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
// The reference at `ref_thunderkitten_alltoall.cu` is the behavioral /
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
//   the upstream ThunderKittens all_to_all.cu — perf must be unchanged)
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
//   Helper kernels for correctness check
// =====================================================================

__global__ void fill_bf16_kernel(__nv_bfloat16 *p, size_t n, __nv_bfloat16 v) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}


// =====================================================================
//   AllToAll — clean communication class (no test/benchmark logic).
//
//   Owns the Broker and exposes only the core ops:
//     zero, fill, all_to_all (scatter=2, gather=1), sync.
// =====================================================================

class AllToAll {
 public:
    AllToAll(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    AllToAll(const AllToAll &) = delete;
    AllToAll &operator=(const AllToAll &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    Broker &broker() { return broker_; }

    void sync() { broker_.sync(); }

    // Zero the local shard of `t` (`bytes` bytes starting at raw_ptrs[rank_]).
    void zero(ParallelTensor &t, size_t bytes) {
        cuda_check(cudaMemsetAsync(t.raw_ptrs[rank_], 0, bytes));
    }

    // Fill the local shard of `t` (`numel` bf16 elements) with `v`.
    void fill(ParallelTensor &t, size_t numel, __nv_bfloat16 v) {
        const int threads = 256;
        const dim3 grid((numel + threads - 1) / threads);
        fill_bf16_kernel<<<grid, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(t.raw_ptrs[rank_]), numel, v);
    }

    // One bracketed all_to_all: barrier → TMA-based all_to_all → barrier.
    //
    //   input:  per-device gl shape (1, N_per_rank, H,         D), bf16
    //   output: per-device gl shape (1, N,          H_per_rank, D), bf16
    //   barrier: per-device int barrier (used by both bracket barriers)
    void all_to_all(ParallelTensor &output, ParallelTensor &input,
                    ParallelTensor &barrier, int N_per_rank, int H, int D) {
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
    double data_size_mb;
    double throughput_gbps;
    double latency_ms;
};

// Benchmark sweep N values. Original upstream uses {16384, 32768, 65536,
// 131072, 262144, 524288}; the largest size allocates ~2 GB per rank
// of input plus another ~2 GB of output. We match the upstream sweep so
// the numbers are directly comparable, but with H/D both 128 to keep the
// per-iter latency in the few-ms range on B200/B300.
static constexpr int kBenchmarkNs[] = {16384, 32768, 65536, 131072, 262144, 524288};
static constexpr int kBenchmarkH    = 128;
static constexpr int kBenchmarkD    = 128;
static constexpr int kNumWarmupIters = 1;
static constexpr int kNumIters       = 5;

// Tiny correctness configuration — small enough that the cudaMemcpy
// readbacks complete instantly.
static constexpr int kCorrectnessN = 256;   // total N (must be world_size * something)
static constexpr int kCorrectnessH = 128;
static constexpr int kCorrectnessD = 128;


// Standalone test/benchmark function — handles correctness + timing.
//
//   * Correctness: each rank fills its full input shard with `rank+1`;
//     after the all_to_all (scatter=2, gather=1), output[rank] should
//     contain a stack of W contributions along axis-1 (depth), with the
//     contribution from source rank `s` carrying the value `s+1`.
//     We sample one cell from each source-rank's slot.
//   * Benchmark: for each N in `Ns`, run `warmup_iters` warmup
//     iterations + `iters` cudaEvent-timed iterations. Bandwidth model
//     follows the upstream reference (per-rank send size = chunk * (W-1),
//     where chunk = (N/W) * (H/W) * D * sizeof(bf16)).
static std::pair<bool, std::vector<MetricRow>> runTest(
    AllToAll &comm,
    const int *Ns,
    int num_sizes,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    const int W = comm.world_size();

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int N = kCorrectnessN;
        const int H = kCorrectnessH;
        const int D = kCorrectnessD;
        const int N_per_rank = N / W;
        const int H_per_rank = H / W;

        const size_t in_numel  = static_cast<size_t>(1) * N_per_rank * H * D;
        const size_t out_numel = static_cast<size_t>(1) * N        * H_per_rank * D;
        const size_t in_bytes  = in_numel  * sizeof(__nv_bfloat16);
        const size_t out_bytes = out_numel * sizeof(__nv_bfloat16);

        ParallelTensor input(comm.broker(), in_bytes,  comm.rank(), W, false);
        ParallelTensor output(comm.broker(), out_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), W, true);

        const __nv_bfloat16 fill_v =
            __float2bfloat16(static_cast<float>(comm.rank() + 1));
        comm.fill(input, in_numel, fill_v);
        comm.zero(output, out_bytes);
        comm.zero(barrier, sizeof(int));
        AllToAll::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        comm.all_to_all(output, input, barrier, N_per_rank, H, D);
        AllToAll::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        // Read back W cells, one per source rank's slot. With scatter=2
        // gather=1, output[rank][b=0, depth=N_per_rank*src + 0,
        // row_block_idx=rank, col_block_idx=0, ...] receives src's data,
        // i.e. (src + 1).
        bool ok = true;
        __nv_bfloat16 host_v;
        const size_t H_p = static_cast<size_t>(H_per_rank);
        const size_t D_s = static_cast<size_t>(D);
        for (int src = 0; src < W; ++src) {
            const size_t depth_idx = static_cast<size_t>(N_per_rank) * src;
            const size_t lin_idx = ((static_cast<size_t>(0) * N + depth_idx)
                                    * H_p + 0) * D_s + 0;
            AllToAll::cuda_check(cudaMemcpy(
                &host_v,
                reinterpret_cast<__nv_bfloat16 *>(output.raw_ptrs[comm.rank()]) + lin_idx,
                sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            const float got = __bfloat162float(host_v);
            const float want = static_cast<float>(src + 1);
            if (std::fabs(got - want) > 1e-2f) { ok = false; break; }
        }
        overall_pass = ok;
        comm.sync();
    }

    // ---- benchmark ----
    const int H = kBenchmarkH;
    const int D = kBenchmarkD;
    const int H_per_rank = H / W;

    for (int i = 0; i < num_sizes; ++i) {
        const int N = Ns[i];
        const int N_per_rank = N / W;

        const size_t in_numel  = static_cast<size_t>(1) * N_per_rank * H * D;
        const size_t out_numel = static_cast<size_t>(1) * N        * H_per_rank * D;
        const size_t in_bytes  = in_numel  * sizeof(__nv_bfloat16);
        const size_t out_bytes = out_numel * sizeof(__nv_bfloat16);

        ParallelTensor input(comm.broker(), in_bytes,  comm.rank(), W, false);
        ParallelTensor output(comm.broker(), out_bytes, comm.rank(), W, false);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), W, true);

        comm.zero(input,  in_bytes);
        comm.zero(output, out_bytes);
        comm.zero(barrier, sizeof(int));
        AllToAll::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            comm.all_to_all(output, input, barrier, N_per_rank, H, D);
        AllToAll::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        AllToAll::cuda_check(cudaEventCreate(&start_evt));
        AllToAll::cuda_check(cudaEventCreate(&stop_evt));
        AllToAll::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            comm.all_to_all(output, input, barrier, N_per_rank, H, D);
        AllToAll::cuda_check(cudaEventRecord(stop_evt));
        AllToAll::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        AllToAll::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        AllToAll::cuda_check(cudaEventDestroy(start_evt));
        AllToAll::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms = total_ms / iters;
        // Per-rank communicated bytes (excluding the local self-chunk):
        // chunk = (N/W) * (H/W) * D * 2 bytes; each rank sends (W-1) chunks.
        const double chunk_bytes = static_cast<double>(N_per_rank)
                                 * static_cast<double>(H_per_rank)
                                 * static_cast<double>(D) * sizeof(__nv_bfloat16);
        const double per_rank_bytes = chunk_bytes * (W - 1);
        const double gbps = avg_ms > 0.0
            ? (per_rank_bytes * 1e-9) / (avg_ms * 1e-3)
            : 0.0;

        rows.push_back({
            static_cast<double>(in_bytes) / 1e6,
            gbps,
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
//   Per-rank child entry point
// =====================================================================

static int rank_main(int rank, int world_size) {
    if (cudaSetDevice(rank) != cudaSuccess) {
        std::fprintf(stderr, "rank %d: cudaSetDevice failed\n", rank);
        return 1;
    }
    cudaFree(0);

    AllToAll comm(rank, world_size);

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
//   main — fork one process per GPU and wait
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
