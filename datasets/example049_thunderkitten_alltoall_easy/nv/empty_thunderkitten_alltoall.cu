// ThunderKittens BF16 multi-GPU AllToAll (NVLink/TMA, scatter=2 gather=1).
// Empty template: complete the two device kernels marked `// TODO` below.
//
// All host scaffolding (per-process fork-and-join, KittensBroker, VMM/IPC
// setup, multicast bring-up for the barrier, correctness check, perf
// benchmark, JSON output) is already wired up — the model only needs to
// fill the device-side bodies.
//
// What you need to implement:
//   * `all_to_all::kernel<SCATTER_AXIS, GATHER_AXIS>` — using a single
//     warp/thread per block, TMA-load one (ROW_BLOCK_SIZE x COL_BLOCK_SIZE)
//     bf16 tile from the local input shard `G.input[G.dev_idx]`, then
//     TMA-store it to the destination device's output shard
//     `G.output[dst_dev_idx]` at the gathered index. Index math follows
//     the upstream pattern: split blockIdx.x into (batch, depth, row_block,
//     col_block); the SCATTER_AXIS picks dst_dev_idx (and trims that
//     coordinate down to the local output range), and GATHER_AXIS shifts
//     the matching input axis by `input.<axis>() * G.dev_idx` to lay each
//     source rank's data into a distinct slab on the destination.
//   * `all_to_all_barrier::kernel` — synchronize all NUM_DEVICES ranks via
//     barrier_all on G.barrier with coord {0} and G.dev_idx (one-liner).

#include "kittens.cuh"
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


// =====================================================================
//   Device-side kernels — fill in the bodies marked `// TODO`
// =====================================================================

namespace all_to_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / globals::COL_BLOCK_SIZE) *
                    (input.rows() / globals::ROW_BLOCK_SIZE) *
                    input.depth() * input.batch());
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

template <int SCATTER_AXIS, int GATHER_AXIS>
__device__ inline void kernel(const globals &G) {
    static_assert(0 <= SCATTER_AXIS && SCATTER_AXIS < 4 && 0 <= GATHER_AXIS && GATHER_AXIS < 4,
        "Scatter and gather axes must be 0, 1, 2, or 3");
    static_assert(SCATTER_AXIS != GATHER_AXIS, "Scatter and gather axes must be different");

    // TODO: allocate one shared_tile from dynamic shared memory using
    //       tma_swizzle_allocator (mirrors the upstream kernel exactly).
    //
    // TODO: derive (batch_idx, depth_idx, row_block_idx, col_block_idx)
    //       by unflattening blockIdx.x against the input layout
    //       (depth * (rows / ROW_BLOCK_SIZE) * (cols / COL_BLOCK_SIZE)).
    //
    // TODO: TMA-load that input tile into shared memory using a single
    //       semaphore (init_semaphore + tma::expect_bytes + tma::load_async
    //       on G.input[G.dev_idx]).
    //
    // TODO: compute dst_dev_idx by splitting the SCATTER_AXIS coordinate
    //       against the *output* extent on that axis, and reduce the
    //       SCATTER_AXIS coordinate modulo that extent so it indexes into
    //       the destination's local shard.
    //
    // TODO: shift the GATHER_AXIS coordinate by `G.input.<axis>() * G.dev_idx`
    //       so the destination collects each source rank's tiles into a
    //       distinct stripe along that axis.
    //
    // TODO: wait on the load semaphore, then tma::store_async the tile
    //       into G.output[dst_dev_idx] at the rewritten coordinate.
}

} // namespace all_to_all

namespace all_to_all_barrier {

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
    static_assert(Config::CLUSTER_SIZE == 1, "Only CLUSTER_SIZE==1 supported here");
    dim3 grid;
    if constexpr (requires { Config::NUM_BLOCKS; })
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();
    dim3 block = dim3{Config::NUM_THREADS, 1, 1};

    int dynsmem = 0;
    if constexpr (requires { Config::DYNAMIC_SHARED_MEMORY; })
        dynsmem = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    else if constexpr (requires { G.dynamic_shared_memory(); })
        dynsmem = static_cast<int>(G.dynamic_shared_memory());

    if (dynsmem > 0) {
        cudaError_t e = cudaFuncSetAttribute(
            reinterpret_cast<const void *>(global_kernel<Config, Globals, Kernel>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynsmem);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("cudaFuncSetAttribute: ") + cudaGetErrorString(e));
    }

    global_kernel<Config, Globals, Kernel><<<grid, block, dynsmem, stream>>>(G);
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

        detail::vmm::vm_alloc_map_set_access(
            &raw_ptrs[local_rank], &allocated_size, bytes, local_rank, local_world_size);

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

        if (multicast) initialize_multicast(broker);
    }

    ParallelTensor(const ParallelTensor &) = delete;
    ParallelTensor &operator=(const ParallelTensor &) = delete;
    ~ParallelTensor() = default;

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
//   Helper kernels for correctness check
// =====================================================================

__global__ void fill_bf16_kernel(__nv_bfloat16 *p, size_t n, __nv_bfloat16 v) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}


// =====================================================================
//   AllToAll — clean communication class (no test/benchmark logic).
// =====================================================================

class AllToAll {
 public:
    AllToAll(int rank, int world_size)
        : rank_(rank), world_size_(world_size), broker_(rank, world_size) {}

    AllToAll(const AllToAll &) = delete;
    AllToAll &operator=(const AllToAll &) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    KittensBroker &broker() { return broker_; }

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

    void all_to_all(ParallelTensor &output, ParallelTensor &input,
                    ParallelTensor &barrier, int N_per_rank, int H, int D) {
        using a2a_layout = all_to_all::globals::parallel_layout;

        const int N_total = N_per_rank * world_size_;
        const int H_per_rank = H / world_size_;

        a2a_layout pgl_in = make_pgl<a2a_layout>(
            reinterpret_cast<uint64_t *>(input.raw_ptrs),
            1, N_per_rank, H, D);
        a2a_layout pgl_out = make_pgl<a2a_layout>(
            reinterpret_cast<uint64_t *>(output.raw_ptrs),
            1, N_total, H_per_rank, D);
        barrier_t<all_to_all_barrier::globals::NUM_DEVICES> pgl_b =
            make_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(
                reinterpret_cast<uint64_t>(barrier.mc_ptr),
                reinterpret_cast<uint64_t *>(barrier.raw_ptrs),
                1, 1, 1, 1);

        all_to_all::globals G_a2a{
            .output = pgl_out, .input = pgl_in, .dev_idx = rank_};
        all_to_all_barrier::globals G_b{.barrier = pgl_b, .dev_idx = rank_};

        launch_kernel<all_to_all_barrier::config,
                      all_to_all_barrier::globals,
                      all_to_all_barrier::kernel>(G_b);
        launch_kernel<all_to_all::config,
                      all_to_all::globals,
                      all_to_all::kernel<2, 1>>(G_a2a);
        launch_kernel<all_to_all_barrier::config,
                      all_to_all_barrier::globals,
                      all_to_all_barrier::kernel>(G_b);
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

static constexpr int kBenchmarkNs[] = {16384, 32768, 65536, 131072, 262144, 524288};
static constexpr int kBenchmarkH    = 128;
static constexpr int kBenchmarkD    = 128;
static constexpr int kNumWarmupIters = 1;
static constexpr int kNumIters       = 5;

static constexpr int kCorrectnessN = 256;
static constexpr int kCorrectnessH = 128;
static constexpr int kCorrectnessD = 128;


static std::pair<bool, std::vector<MetricRow>> runTest(
    AllToAll &comm,
    const int *Ns,
    int num_sizes,
    int warmup_iters,
    int iters) {

    std::vector<MetricRow> rows;
    rows.reserve(num_sizes);

    const int W = comm.world_size();

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
//   JSON output
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
