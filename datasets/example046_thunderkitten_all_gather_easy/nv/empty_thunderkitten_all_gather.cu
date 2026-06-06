// ThunderKittens BF16 multi-GPU AllGather (NVLink/multicast).
// Empty template: complete the two device kernels marked `// TODO` below.
//
// All host scaffolding (per-process fork-and-join, KittensBroker, VMM/IPC
// setup, multicast bring-up, correctness check, perf benchmark, JSON output)
// is already wired up — the model only needs to fill the device-side bodies.
//
// What you need to implement:
//   * `all_gather::kernel` — single-threaded TMA-driven block. Allocate a
//     `globals::shared_tile` (128x128 bf16) in dynamic shared memory, use
//     `tma::load_async` to pull this rank's (row_block_idx, col_block_idx)
//     tile from `G.input[G.dev_idx]`, wait on the arrival semaphore, then
//     `tma::store_async` it through the multicast output pointer `G.output`
//     at column block `col_blocks_per_dev * G.dev_idx + col_block_idx` so
//     every peer receives the tile at the correct stripe of the gathered
//     output.
//   * `all_gather_barrier::kernel` — synchronize all NUM_DEVICES ranks via
//     `barrier_all` on G.barrier.

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
//   Device-side kernels — fill in the bodies marked `// TODO`
// =====================================================================

namespace all_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int BLOCK_SIZE = 128;

    using shared_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using output_layout = pgl<gl<bf16, 1, 1, -1, -1>, NUM_DEVICES, true, shared_tile>;
    using input_layout = pgl<gl<bf16, 1, 1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    output_layout output;
    input_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / BLOCK_SIZE), (input.rows() / BLOCK_SIZE));
    }

    __host__ inline int dynamic_shared_memory() const {
        return sizeof(shared_tile) + 1024;
    }
};

__device__ inline void kernel(const globals &G) {
    // TODO: single-threaded TMA-driven block.
    //
    // 1. Allocate a `globals::shared_tile` (128x128 bf16) in dynamic shared
    //    memory using `tma_swizzle_allocator` over `extern __shared__ int __shm[]`.
    // 2. Compute the tile coordinates:
    //      - row_block_idx       = blockIdx.y
    //      - col_block_idx       = blockIdx.x
    //      - col_blocks_per_dev  = G.output.cols() / BLOCK_SIZE / NUM_DEVICES
    // 3. Initialize a __shared__ `semaphore arrived` and call
    //    `tma::expect_bytes(arrived, sizeof(tile))`.
    // 4. Issue `tma::load_async` from this rank's input shard
    //      G.input[G.dev_idx] at coords {row_block_idx, col_block_idx}
    //    into the shared tile, then `wait(arrived, 0)`.
    // 5. Issue `tma::store_async` through the multicast output pointer
    //    `G.output` at coords
    //      {row_block_idx, col_blocks_per_dev * G.dev_idx + col_block_idx}
    //    so each rank's slice ends up in the correct column stripe of every
    //    peer's gathered output.
}

} // namespace all_gather

namespace all_gather_barrier {

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
    static_assert(Config::CLUSTER_SIZE == 1, "Only CLUSTER_SIZE==1 supported here");
    dim3 grid;
    if constexpr (requires { Config::NUM_BLOCKS; })
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();
    dim3 block = dim3{Config::NUM_THREADS, 1, 1};

    int smem = 0;
    if constexpr (requires { G.dynamic_shared_memory(); })
        smem = G.dynamic_shared_memory();
    else if constexpr (requires { Config::DYNAMIC_SHARED_MEMORY; })
        smem = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);

    if (smem > 0) {
        auto attr_err = cudaFuncSetAttribute(
            (void *)global_kernel<Config, Globals, Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (attr_err != cudaSuccess)
            throw std::runtime_error(std::string("cudaFuncSetAttribute: ")
                                     + cudaGetErrorString(attr_err));
    }

    global_kernel<Config, Globals, Kernel><<<grid, block, smem, stream>>>(G);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("kernel launch: ") + cudaGetErrorString(err));
}


// =====================================================================
//   ParallelTensor — pure-C++ replacement for kittens::py::TKParallelTensor
//
//   Uses driver-API VMM allocation, swaps POSIX FDs through KittensBroker
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
//   AllGather — clean communication class (no test/benchmark logic).
//
//   Owns the KittensBroker and exposes only the core ops:
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

    // One bracketed all-gather: barrier → TMA gather → barrier.
    //
    //   * `output`  shape (N, N), multicast=true
    //   * `input`   shape (N, N/W), per-rank shard (multicast=false)
    //   * `barrier` 1-element int, multicast=true
    void all_gather(ParallelTensor &output,
                    ParallelTensor &input,
                    ParallelTensor &barrier,
                    int N) {
        const int W = world_size_;
        using out_layout = all_gather::globals::output_layout;
        using in_layout  = all_gather::globals::input_layout;

        out_layout pgl_o = make_pgl<out_layout>(
            reinterpret_cast<uint64_t>(output.mc_ptr),
            reinterpret_cast<uint64_t *>(output.raw_ptrs),
            1, 1, N, N);
        in_layout pgl_i = make_pgl<in_layout>(
            reinterpret_cast<uint64_t *>(input.raw_ptrs),
            1, 1, N, N / W);
        barrier_t<all_gather_barrier::globals::NUM_DEVICES> pgl_b =
            make_pgl<barrier_t<all_gather_barrier::globals::NUM_DEVICES>>(
                reinterpret_cast<uint64_t>(barrier.mc_ptr),
                reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
                1, 1, 1, 1);

        all_gather::globals G_ag{
            .output = pgl_o,
            .input = pgl_i,
            .dev_idx = rank_,
        };
        all_gather_barrier::globals G_b{.barrier = pgl_b, .dev_idx = rank_};

        launch_kernel<all_gather_barrier::config,
                      all_gather_barrier::globals,
                      all_gather_barrier::kernel>(G_b);
        launch_kernel<all_gather::config,
                      all_gather::globals,
                      all_gather::kernel>(G_ag);
        launch_kernel<all_gather_barrier::config,
                      all_gather_barrier::globals,
                      all_gather_barrier::kernel>(G_b);
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
