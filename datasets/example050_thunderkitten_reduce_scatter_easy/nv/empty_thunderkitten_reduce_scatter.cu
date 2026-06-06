// ThunderKittens BF16 multi-GPU ReduceScatterSum (NVLink/multicast).
//
// Self-contained C++/CUDA benchmark — no PyTorch, no pybind, no torchrun.
// `int main()` forks NUM_DEVICES (8) child processes, each pinned to one GPU.
// Each child uses the ThunderKittens VMM/IPC helpers (CUDA driver API)
// together with a POSIX-socket-backed KittensBroker to allocate shareable
// device memory, exchange handles, and bring up an NVLink multicast region.
//
// Code layout (per the dataset readme guidelines):
//   * Device-side kernels — `reduce_scatter::kernel` (multimem.ld_reduce on
//     the multicast input pointer, plain stg to the local output) and
//     `reduce_scatter_barrier::kernel` (multimem.red bracket barrier).
//   * `ParallelTensor` — pure-C++ replacement for kittens::py::TKParallelTensor;
//     allocates VMM-backed shareable physical memory, exchanges POSIX fds,
//     and (optionally) lays a multicast handle on top.
//   * `ReduceScatter` — clean communication class. Owns the KittensBroker
//     and exposes only the core ops (zero / fill / reduce_scatter /
//     verify_filled / sync). No test or benchmark logic embedded.
//   * `runTest(...)` — dedicated standalone function. Runs the correctness
//     check at N=kCorrectnessN and the latency/throughput benchmark across
//     kBenchmarkSizes (1 warmup + kNumIters timed iterations per size,
//     cudaEvent timed).
//   * `printJsonResult(...)` — emits exactly one JSON object on rank 0.
//   * `rank_main` / `main` — fork NUM_DEVICES children, wait, and propagate
//     a non-zero exit if any rank failed correctness or crashed.

#include "kittens.cuh"
#include "pyutils/broker.cuh"          // KittensBroker (POSIX SHM + Unix sockets, no torch)
#include "types/system/vmm.cuh"        // CUDA driver VMM (cuMemCreate / cuMulticast*)
#include "types/system/ipc.cuh"        // FD-based IPC handle export/import

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


// =====================================================================
//   Device-side kernels (kernel bodies are intentionally identical to
//   the upstream ThunderKittens reduce_scatter.cu — perf must be unchanged)
// =====================================================================

namespace reduce_scatter {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 2;
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using input_layout  = pgl<gl<bf16, 1, -1, -1, -1>, NUM_DEVICES, true>;
    using output_layout = pgl<gl<bf16, 1, -1, -1, -1>, NUM_DEVICES, false>;

    output_layout output;
    input_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(output.cols() / NUM_ELEMS_PER_BLOCK * output.rows(), output.depth());
    }
};

__device__ inline void kernel(const globals &G) {
    // TODO
}

} // namespace reduce_scatter

namespace reduce_scatter_barrier {

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
    // TODO
}

} // namespace reduce_scatter_barrier


// =====================================================================
//   Minimal launch helper (no cluster, no PDL, no torch stream)
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
    static_assert(Config::CLUSTER_SIZE == 1, "Only CLUSTER_SIZE==1 supported here");
    dim3 grid;
    if constexpr (requires { Config::NUM_BLOCKS; })
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();
    dim3 block = dim3{Config::NUM_THREADS, 1, 1};
    global_kernel<Config, Globals, Kernel><<<grid, block, 0, stream>>>(G);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("kernel launch: ") + cudaGetErrorString(err));
}


// =====================================================================
//   ParallelTensor — pure-C++ replacement for kittens::py::TKParallelTensor
//
//   Uses driver-API VMM allocation, swaps POSIX FDs through KittensBroker
//   to grant peer access on every rank, and (optionally) lays a multicast
//   handle on top so the kernel can use mc_ptr.
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

        // 1. Allocate VMM-backed shareable physical memory and map it.
        detail::vmm::vm_alloc_map_set_access(
            &raw_ptrs[local_rank], &allocated_size, bytes, local_rank, local_world_size);

        // 2. Exchange POSIX fds so every rank can map every other rank's pages.
        using handle_t = detail::ipc::handle<detail::ipc::flavor::VMM>;
        handle_t my_ipc;
        detail::ipc::export_handle(&my_ipc, raw_ptrs[local_rank]);

        std::vector<int> all_fds(local_world_size, -1);
        broker.exchange_fds(all_fds.data(), my_ipc.handle_);

        for (int i = 0; i < local_world_size; ++i) {
            if (i == local_rank) continue;
            handle_t peer;
            peer.handle_ = all_fds[i];
            detail::ipc::import_handle<handle_t>(&raw_ptrs[i], peer, allocated_size, local_world_size);
        }

        // 3. (Optional) Bring up a multicast handle for `mc_ptr`-style kernels.
        if (multicast) initialize_multicast(broker);
    }

    ParallelTensor(const ParallelTensor &) = delete;
    ParallelTensor &operator=(const ParallelTensor &) = delete;

    ~ParallelTensor() = default;  // process exit reclaims everything

    void initialize_multicast(KittensBroker &broker) {
        using handle_t = detail::ipc::handle<detail::ipc::flavor::VMM>;

        detail::vmm::multicast_check(local_rank);
        detail::ipc::check_support(local_rank);
        detail::vmm::handle multicast_handle;

        if (local_rank == 0) {
            detail::vmm::multicast_create_handle(
                &multicast_handle, &mc_allocated_size, allocated_size, local_world_size);
            if (allocated_size != mc_allocated_size)
                throw std::runtime_error("multicast allocated size != memory allocated size");
            handle_t ipc_handle;
            detail::ipc::export_handle(&ipc_handle, multicast_handle);
            broker.broadcast_fd(nullptr, ipc_handle.handle_, 0);
        } else {
            handle_t ipc_handle;
            broker.broadcast_fd(&ipc_handle.handle_, -1, 0);
            mc_allocated_size = allocated_size;
            detail::ipc::import_handle<handle_t>(
                &multicast_handle, ipc_handle, mc_allocated_size, local_world_size);
        }

        detail::vmm::multicast_bind_device(multicast_handle, local_rank);
        broker.sync();

        detail::vmm::handle memory_handle;
        detail::vmm::vm_retrieve_handle(&memory_handle, raw_ptrs[local_rank]);
        detail::vmm::multicast_bind_memory(multicast_handle, memory_handle, allocated_size);
        broker.sync();

        detail::vmm::vm_map(&mc_ptr, multicast_handle, mc_allocated_size);
        detail::vmm::vm_set_access(mc_ptr, mc_allocated_size, local_world_size);

        detail::vmm::vm_free(multicast_handle);
        detail::vmm::vm_free(memory_handle);
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
//   ReduceScatter — clean communication class (no test/benchmark logic).
//
//   Owns the KittensBroker and exposes only the core ops:
//     zero, fill, reduce_scatter, verify_filled, sync.
// =====================================================================

class ReduceScatter {
 public:
    ReduceScatter(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    ReduceScatter(const ReduceScatter &) = delete;
    ReduceScatter &operator=(const ReduceScatter &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    KittensBroker &broker() { return broker_; }

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

    // One bracketed reduce-scatter: barrier → multimem reduce + local store → barrier.
    //
    // input  : (N, N) per-rank replica with multicast handle, summed across ranks
    // output : (N, N/world_size) per-rank shard (rank r receives shard r)
    void reduce_scatter(ParallelTensor &output, ParallelTensor &input,
                        ParallelTensor &barrier, int N) {
        using in_layout  = reduce_scatter::globals::input_layout;
        using out_layout = reduce_scatter::globals::output_layout;
        in_layout pgl_in = make_pgl<in_layout>(
            reinterpret_cast<uint64_t>(input.mc_ptr),
            reinterpret_cast<uint64_t *>(input.raw_ptrs),
            1, 1, N, N);
        out_layout pgl_out = make_pgl<out_layout>(
            reinterpret_cast<uint64_t *>(output.raw_ptrs),
            1, 1, N, N / world_size_);
        barrier_t<reduce_scatter_barrier::globals::NUM_DEVICES> pgl_b =
            make_pgl<barrier_t<reduce_scatter_barrier::globals::NUM_DEVICES>>(
                reinterpret_cast<uint64_t>(barrier.mc_ptr),
                reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
                1, 1, 1, 1);

        reduce_scatter::globals G_rs{.output = pgl_out, .input = pgl_in, .dev_idx = rank_};
        reduce_scatter_barrier::globals G_b{.barrier = pgl_b, .dev_idx = rank_};

        launch_kernel<reduce_scatter_barrier::config,
                      reduce_scatter_barrier::globals,
                      reduce_scatter_barrier::kernel>(G_b);
        launch_kernel<reduce_scatter::config,
                      reduce_scatter::globals,
                      reduce_scatter::kernel>(G_rs);
        launch_kernel<reduce_scatter_barrier::config,
                      reduce_scatter_barrier::globals,
                      reduce_scatter_barrier::kernel>(G_b);
    }

    // Sample-check: read the first and last bf16 element of the local shard
    // and confirm both are within 1e-2 of `expected`.
    bool verify_filled(ParallelTensor &t, size_t numel, float expected) {
        __nv_bfloat16 host[2] = {};
        cuda_check(cudaMemcpy(&host[0], t.raw_ptrs[rank_],
                              sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(&host[1],
                              reinterpret_cast<__nv_bfloat16 *>(t.raw_ptrs[rank_]) + (numel - 1),
                              sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        float v0 = __bfloat162float(host[0]);
        float v1 = __bfloat162float(host[1]);
        return std::fabs(v0 - expected) < 1e-2f && std::fabs(v1 - expected) < 1e-2f;
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
    double data_size_mb;
    double throughput_gbps;
    double latency_ms;
};

static constexpr int kBenchmarkSizes[] = {2048, 4096, 8192, 16384, 32768, 65536};
static constexpr int kNumWarmupIters  = 1;
static constexpr int kNumIters        = 5;
static constexpr int kCorrectnessN    = 2048;

// Standalone test/benchmark function — handles correctness + timing.
//
//   * Correctness: each rank fills its full input replica with `rank+1`;
//     after the reduce-scatter every output element on rank r should equal
//     sum(1..world_size) (the same shard sum on every rank).
//   * Benchmark: for each N in `sizes`, run `warmup_iters` warmup
//     iterations + `iters` cudaEvent-timed iterations. Bandwidth model
//     follows the standard NCCL reduce-scatter bus formula:
//     bytes_moved = input_bytes * (world_size - 1) / world_size.
static std::pair<bool, std::vector<MetricRow>> runTest(
    ReduceScatter &comm,
    const int *sizes,
    int num_sizes,
    int correctness_n,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    // ---- correctness ----
    bool overall_pass = false;
    {
        const int N = correctness_n;
        const size_t input_numel  = static_cast<size_t>(N) * N;
        const size_t input_bytes  = input_numel * sizeof(__nv_bfloat16);
        const size_t output_numel = static_cast<size_t>(N) * (N / comm.world_size());
        const size_t output_bytes = output_numel * sizeof(__nv_bfloat16);

        ParallelTensor input(comm.broker(), input_bytes, comm.rank(), comm.world_size(), true);
        ParallelTensor output(comm.broker(), output_bytes, comm.rank(), comm.world_size(), false);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), comm.world_size(), true);

        const __nv_bfloat16 fill_v = __float2bfloat16(static_cast<float>(comm.rank() + 1));
        comm.fill(input, input_numel, fill_v);
        comm.zero(output, output_bytes);
        comm.zero(barrier, sizeof(int));
        ReduceScatter::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        comm.reduce_scatter(output, input, barrier, N);
        ReduceScatter::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        const float expected =
            static_cast<float>(comm.world_size() * (comm.world_size() + 1) / 2);
        overall_pass = comm.verify_filled(output, output_numel, expected);
        comm.sync();
    }

    // ---- benchmark ----
    for (int i = 0; i < num_sizes; ++i) {
        const int N = sizes[i];
        const size_t input_numel  = static_cast<size_t>(N) * N;
        const size_t input_bytes  = input_numel * sizeof(__nv_bfloat16);
        const size_t output_bytes = static_cast<size_t>(N) * (N / comm.world_size())
                                    * sizeof(__nv_bfloat16);

        ParallelTensor input(comm.broker(), input_bytes, comm.rank(), comm.world_size(), true);
        ParallelTensor output(comm.broker(), output_bytes, comm.rank(), comm.world_size(), false);
        ParallelTensor barrier(comm.broker(), sizeof(int), comm.rank(), comm.world_size(), true);

        comm.zero(input, input_bytes);
        comm.zero(output, output_bytes);
        comm.zero(barrier, sizeof(int));
        ReduceScatter::cuda_check(cudaDeviceSynchronize());
        comm.sync();

        for (int w = 0; w < warmup_iters; ++w)
            comm.reduce_scatter(output, input, barrier, N);
        ReduceScatter::cuda_check(cudaDeviceSynchronize());

        cudaEvent_t start_evt, stop_evt;
        ReduceScatter::cuda_check(cudaEventCreate(&start_evt));
        ReduceScatter::cuda_check(cudaEventCreate(&stop_evt));
        ReduceScatter::cuda_check(cudaEventRecord(start_evt));
        for (int it = 0; it < iters; ++it)
            comm.reduce_scatter(output, input, barrier, N);
        ReduceScatter::cuda_check(cudaEventRecord(stop_evt));
        ReduceScatter::cuda_check(cudaEventSynchronize(stop_evt));

        float total_ms = 0.0f;
        ReduceScatter::cuda_check(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
        ReduceScatter::cuda_check(cudaEventDestroy(start_evt));
        ReduceScatter::cuda_check(cudaEventDestroy(stop_evt));

        const double avg_ms = total_ms / iters;
        const double bytes_moved = static_cast<double>(input_bytes)
                                 * (comm.world_size() - 1) / comm.world_size();
        const double gbps = avg_ms > 0.0
            ? (bytes_moved * 1e-9) / (avg_ms * 1e-3)
            : 0.0;

        rows.push_back({
            static_cast<double>(input_bytes) / 1e6,
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

    ReduceScatter comm(rank, world_size);

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
    constexpr int WORLD_SIZE = reduce_scatter::globals::NUM_DEVICES;

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
