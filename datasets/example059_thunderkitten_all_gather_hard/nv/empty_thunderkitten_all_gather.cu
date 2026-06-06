// HARD task — multi-GPU all_gather from scratch (no ThunderKittens).
//
// Behavioral spec (mirrors the reference at `ref_thunderkitten_all_gather.cu`):
// ThunderKittens BF16 multi-GPU AllGather (NVLink/multicast).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// `int main()` forks NUM_DEVICES (8) child processes, each pinned to one GPU.
// Each child uses the ThunderKittens VMM/IPC helpers (CUDA driver API)
// together with a POSIX-socket-backed Broker to allocate shareable
// device memory, exchange handles, and bring up an NVLink multicast region
// for the gathered output.
//
// Code layout (per the dataset readme guidelines):
//   * Device-side kernels — `all_gather::kernel` (single-threaded TMA: load
//     a 128x128 bf16 tile from this rank's input shard into shared memory,
//     then TMA-store it through the multicast output pointer to the right
//     column slice on every rank) and `all_gather_barrier::kernel`
//     (multimem.red bracket barrier).
//   * `ParallelTensor` — pure-C++ replacement for TKParallelTensor;
//     allocates VMM-backed shareable physical memory, exchanges POSIX fds,
//     and (optionally) lays a multicast handle on top.
//   * `AllGather` — clean communication class. Owns the Broker and
//     exposes only the core ops (zero / fill / all_gather / verify_gather /
//     sync). No test or benchmark logic embedded.
//   * `runTest(...)` — dedicated standalone function. Runs the correctness
//     check at N=kCorrectnessN and the latency/throughput benchmark across
//     kBenchmarkSizes (1 warmup + kNumIters timed iterations per size,
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
// The reference at `ref_thunderkitten_all_gather.cu` is the behavioral /
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
//   the upstream ThunderKittens all_gather.cu — perf must be unchanged)
// =====================================================================

namespace all_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 1;

    // TODO: define remaining kernel-launch shape parameters
    //       (NUM_THREADS, DYNAMIC_SHARED_MEMORY, etc.).
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int BLOCK_SIZE = 128;

    // TODO: define the kernel's globals (TMA descriptors,
    //       peer-pointer arrays, dev_idx, runtime params, plus
    //       any host-side helpers used by the launcher such as
    //       `dim3 grid()` / `int dynamic_shared_memory()`).
};

__device__ inline void kernel(const globals &G) {
    // TODO
}

} // namespace all_gather

namespace all_gather_barrier {

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

} // namespace all_gather_barrier


// =====================================================================
//   Minimal launch helper (no cluster, no PDL, no torch stream).
//
//   Picks dynamic shared memory from the kernel's Globals (if it provides
//   `dynamic_shared_memory()`) or otherwise from the Config's
//   DYNAMIC_SHARED_MEMORY constant. The all_gather kernel needs ~33 KB
//   of dynamic shared memory for the 128x128 bf16 TMA tile.
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
//
//   Uses driver-API VMM allocation, swaps POSIX FDs through Broker
//   to grant peer access on every rank, and (optionally) lays a multicast
//   handle on top so kernels can use mc_ptr.
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

    ~ParallelTensor() = default;  // process exit reclaims everything

    void initialize_multicast(Broker &broker) {
        // TODO: implement (uses cuMemCreate / cuMemMap /
        //       cuMemSetAccess + POSIX-FD-passing IPC + optional
        //       cuMulticast* in the reference).
    }
};


// =====================================================================
//   Init kernel for the correctness check
// =====================================================================

__global__ void fill_bf16_kernel(__nv_bfloat16 *p, size_t n, __nv_bfloat16 v) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}


// =====================================================================
//   AllGather — clean communication class (no test/benchmark logic).
//
//   Owns the Broker and exposes only the core ops:
//     zero, fill, all_gather, verify_gather, sync.
// =====================================================================

class AllGather {
 public:
    AllGather(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    AllGather(const AllGather &) = delete;
    AllGather &operator=(const AllGather &) = delete;

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

    // One bracketed all-gather: barrier → TMA gather → barrier.
    //
    //   * `output`  shape (N, N), multicast=true
    //   * `input`   shape (N, N/W), per-rank shard (multicast=false)
    //   * `barrier` 1-element int, multicast=true
    void all_gather(ParallelTensor &output,
                    ParallelTensor &input,
                    ParallelTensor &barrier,
                    int N) {
        // TODO: implement using your kernel + Broker + ParallelTensor.
        //       (See `ref_thunderkitten_*.cu` for the upstream version.)
    }

    // Sample-check: each rank `k` filled its input with (k+1). After the
    // gather, on every rank `output[0, k * N/W]` should equal (k+1) for
    // every k in [0, W). Returns true iff all W samples match.
    bool verify_gather(ParallelTensor &output, int N) {
        const int W = world_size_;
        const size_t cols = static_cast<size_t>(N);  // output.cols
        std::vector<__nv_bfloat16> samples(W);
        for (int k = 0; k < W; ++k) {
            const size_t idx = static_cast<size_t>(k) * (cols / W);  // row 0
            cuda_check(cudaMemcpy(&samples[k],
                                  reinterpret_cast<__nv_bfloat16 *>(output.raw_ptrs[rank_]) + idx,
                                  sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        }
        for (int k = 0; k < W; ++k) {
            float v = __bfloat162float(samples[k]);
            if (std::fabs(v - static_cast<float>(k + 1)) > 1e-2f) return false;
        }
        return true;
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

// Sizes mirror the upstream ThunderKittens benchmark.py:
//   for N in [4096, 8192, 16384, 32768, 65536]: run(N, ...)
// Output is N×N bf16, input shard is N×N/W bf16.
static constexpr int kBenchmarkSizes[] = {4096, 8192, 16384, 32768, 65536};
static constexpr int kNumWarmupIters  = 1;
static constexpr int kNumIters        = 5;
static constexpr int kCorrectnessN    = 4096;

// Standalone test/benchmark function — handles correctness + timing.
//
//   * Correctness: each rank fills its (N × N/W) input shard with `rank+1`;
//     after the all-gather, every rank's output should be a column-striped
//     matrix where columns [k*N/W, (k+1)*N/W) hold (k+1).
//   * Benchmark: for each N in `sizes`, run `warmup_iters` warmup
//     iterations + `iters` cudaEvent-timed iterations. Bandwidth uses the
//     standard NCCL all-gather convention (per-rank traffic = (W-1)/W of
//     the gathered output).
static std::pair<bool, std::vector<MetricRow>> runTest(
    AllGather &comm,
    const int *sizes,
    int num_sizes,
    int correctness_n,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    const int W = comm.world_size();

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int N = correctness_n;
        const size_t in_numel  = static_cast<size_t>(N) * (N / W);
        const size_t in_bytes  = in_numel  * sizeof(__nv_bfloat16);
        const size_t out_numel = static_cast<size_t>(N) * N;
        const size_t out_bytes = out_numel * sizeof(__nv_bfloat16);

        ParallelTensor input(comm.broker(), in_bytes, comm.rank(), comm.world_size(), false);
        ParallelTensor output(comm.broker(), out_bytes, comm.rank(), comm.world_size(), true);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), comm.world_size(), true);

        const __nv_bfloat16 fill_v = __float2bfloat16(static_cast<float>(comm.rank() + 1));
        comm.fill(input, in_numel, fill_v);
        comm.zero(output, out_bytes);
        comm.zero(barrier, sizeof(int));
        AllGather::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        comm.all_gather(output, input, barrier, N);
        AllGather::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        overall_pass = comm.verify_gather(output, N);
        comm.sync();
    }

    // ---- benchmark ----
    for (int i = 0; i < num_sizes; ++i) {
        const int N = sizes[i];
        const size_t in_bytes  = static_cast<size_t>(N) * (N / W) * sizeof(__nv_bfloat16);
        const size_t out_bytes = static_cast<size_t>(N) * N * sizeof(__nv_bfloat16);

        ParallelTensor input(comm.broker(), in_bytes, comm.rank(), comm.world_size(), false);
        ParallelTensor output(comm.broker(), out_bytes, comm.rank(), comm.world_size(), true);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), comm.world_size(), true);

        comm.zero(input, in_bytes);
        comm.zero(output, out_bytes);
        comm.zero(barrier, sizeof(int));
        AllGather::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            comm.all_gather(output, input, barrier, N);
        AllGather::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        AllGather::cuda_check(cudaEventCreate(&start_evt));
        AllGather::cuda_check(cudaEventCreate(&stop_evt));
        AllGather::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            comm.all_gather(output, input, barrier, N);
        AllGather::cuda_check(cudaEventRecord(stop_evt));
        AllGather::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        AllGather::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        AllGather::cuda_check(cudaEventDestroy(start_evt));
        AllGather::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms = total_ms / iters;
        // NCCL all-gather convention: per-rank bytes moved = (W-1)/W * total output.
        const double bytes_moved = static_cast<double>(out_bytes)
                                 * (W - 1) / W;
        const double gbps = avg_ms > 0.0
            ? (bytes_moved * 1e-9) / (avg_ms * 1e-3)
            : 0.0;

        rows.push_back({
            static_cast<double>(out_bytes) / 1e6,
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
    cudaFree(0);  // force runtime+driver context creation

    AllGather comm(rank, world_size);

    auto [correctness, rows] = runTest(
        comm,
        kBenchmarkSizes,
        static_cast<int>(sizeof(kBenchmarkSizes) / sizeof(kBenchmarkSizes[0])),
        kCorrectnessN,
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
    constexpr int WORLD_SIZE = all_gather::globals::NUM_DEVICES;

    // Wipe any stale broker SHM left over from a crashed prior invocation.
    shm_unlink("/kittens_broker_shm");

    pid_t pids[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (pid == 0) {
            // Die with the parent so we never leave orphan ranks holding
            // multicast / broker resources if the parent is killed (e.g. by
            // build_and_run.py's subprocess timeout).
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

    // A correctness mismatch is reported via JSON ("Correctness": "FAIL");
    // exit non-zero so the build_and_run.py harness treats the run as failed.
    if (any_correctness_fail) overall_rc = 1;
    return overall_rc;
}
