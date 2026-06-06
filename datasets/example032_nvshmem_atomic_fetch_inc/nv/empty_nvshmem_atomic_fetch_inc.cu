/*
 * NVSHMEM Atomic Fetch-Inc Benchmark
 *
 * Sweeps over the number of consecutive nvshmem_uint64_atomic_fetch_inc
 * calls per PE per round (ops_per_pe: 1, 2, 4, 8, 16, 32, 64) and
 * measures:
 *   - latency_avg : average per-op round-trip latency (us)
 *   - throughput_avg : aggregate system throughput (Mops/s)
 *
 * All PEs contend on a single counter on PE 0.  Correctness is verified
 * by confirming every ticket slot in the result buffer is filled with a
 * valid PE id.
 *
 * Output: Exactly ONE JSON object on stdout.  All diagnostics go to stderr.
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int kWarmupIters             = 5;
static constexpr int kBenchIters              = 50;
static constexpr int kMaxOpsPerPe             = 10240;
static constexpr int kOpsPerPeSizes[]         = {1, 4, 16, 64, 256, 640, 1280, 2560, 5120, 10240};
static constexpr int kNumSizes                =
    static_cast<int>(sizeof(kOpsPerPeSizes) / sizeof(kOpsPerPeSizes[0]));

// ── CUDA error checking ───────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ── Kernels ───────────────────────────────────────────────────────────────────

// Each PE calls this with ops_count consecutive atomic_fetch_incs.
// total_cycles_out captures the wall time for all ops_count calls.
__global__ void kernel_atomic_fetch_inc_batch(uint64_t *global_counter,
                                              uint64_t *shared_buf,
                                              int       my_pe,
                                              int       ops_count,
                                              uint64_t *total_cycles_out)
{
    uint64_t t0 = clock64();
    // TODO
    *total_cycles_out = clock64() - t0;
}

// PE 0 resets the shared counter and the first total_slots entries of shared_buf.
__global__ void kernel_reset_shared(uint64_t *counter, uint64_t *buf, int total_slots)
{
    *counter = 0;
    for (int i = threadIdx.x; i < total_slots; i += blockDim.x)
        buf[i] = UINT64_MAX;
}

// ── NvshmemAtomicBenchmark ────────────────────────────────────────────────────

class NvshmemAtomicBenchmark {
public:
    NvshmemAtomicBenchmark(int my_pe, int num_pes)
        : my_pe_(my_pe), num_pes_(num_pes) {}

    ~NvshmemAtomicBenchmark() { freeBuffers(); }

    NvshmemAtomicBenchmark(const NvshmemAtomicBenchmark &) = delete;
    NvshmemAtomicBenchmark &operator=(const NvshmemAtomicBenchmark &) = delete;

    void allocate() {
         // TODO
    }

    void freeBuffers() {
        if (global_counter_) { nvshmem_free(global_counter_); global_counter_ = nullptr; }
        if (shared_buf_)      { nvshmem_free(shared_buf_);     shared_buf_      = nullptr; }
        if (total_cycles_)    { nvshmem_free(total_cycles_);   total_cycles_    = nullptr; }
    }

    // PE 0 resets the shared counter and first total_slots entries of shared_buf.
    // Call barrier() before any PE proceeds after this.
    void resetShared(int total_slots) {
        if (my_pe_ == 0) {
            int threads = std::min(total_slots, 1024);
            kernel_reset_shared<<<1, threads>>>(global_counter_, shared_buf_, total_slots);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // All PEs call concurrently after barrier().
    // Runs ops_count atomic_fetch_incs and returns total elapsed time in us.
    double measureBatch(int ops_count, int clock_khz) {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        CUDA_CHECK(cudaMemset(total_cycles_, 0, sizeof(uint64_t)));
        kernel_atomic_fetch_inc_batch<<<1, 1>>>(
            global_counter_, shared_buf_, my_pe_, ops_count, total_cycles_);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_quiet();
        uint64_t cycles = 0;
        CUDA_CHECK(cudaMemcpy(&cycles, total_cycles_,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return static_cast<double>(cycles)
               / (static_cast<double>(clock_khz) * 1e3) * 1e6;
    }

    // PE 0 verifies that the first total_slots entries of shared_buf
    // are all filled with valid PE ids (not UINT64_MAX).
    bool verify(int total_slots) const {
        std::vector<uint64_t> host(total_slots);
        CUDA_CHECK(cudaSetDevice(my_pe_));
        CUDA_CHECK(cudaMemcpy(host.data(), shared_buf_,
                              total_slots * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < total_slots; ++i) {
            if (host[i] == UINT64_MAX || host[i] >= static_cast<uint64_t>(num_pes_))
                return false;
        }
        return true;
    }

    // MPI allreduce to get the average of a per-PE double.
    double allreduceAvg(double local_val) const {
        double sum = 0.0;
        MPI_Allreduce(&local_val, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return sum / num_pes_;
    }

    void barrier() const { nvshmem_barrier_all(); }

    int  myPe()   const { return my_pe_;      }
    int  numPes() const { return num_pes_;     }
    bool isRoot() const { return my_pe_ == 0; }

private:
    int       my_pe_;
    int       num_pes_;
    uint64_t *global_counter_ = nullptr;
    uint64_t *shared_buf_     = nullptr;
    uint64_t *total_cycles_   = nullptr;
};

// ── runTest ───────────────────────────────────────────────────────────────────

struct MetricEntry {
    int    ops_per_pe;
    double latency_avg;    // us per op
    double throughput_avg; // Mops/s aggregate
    bool   pass;
};

static void runTest(NvshmemAtomicBenchmark &bench, int clock_khz,
                    int warmup_iters, int bench_iters)
{
    bool overall_pass = true;
    std::vector<MetricEntry> rows;

    for (int si = 0; si < kNumSizes; ++si) {
        const int ops_per_pe   = kOpsPerPeSizes[si];
        const int total_slots  = bench.numPes() * ops_per_pe;

        // Warmup
        for (int w = 0; w < warmup_iters; ++w) {
            bench.resetShared(total_slots);
            bench.barrier();
            bench.measureBatch(ops_per_pe, clock_khz);
            bench.barrier();
        }

        // Timed iterations
        double total_batch_us = 0.0;
        bool   size_pass      = true;
        for (int it = 0; it < bench_iters; ++it) {
            bench.resetShared(total_slots);
            bench.barrier();
            double batch_us = bench.measureBatch(ops_per_pe, clock_khz);
            bench.barrier();

            // Use max across PEs (bottleneck time) for throughput
            double avg_batch_us = bench.allreduceAvg(batch_us);
            total_batch_us += avg_batch_us;

            if (bench.isRoot() && !bench.verify(total_slots))
                size_pass = false;
            int p = size_pass ? 1 : 0;
            MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
            size_pass = (p != 0);
        }
        overall_pass = overall_pass && size_pass;

        if (bench.isRoot()) {
            double avg_batch_us   = total_batch_us / bench_iters;
            // per-op latency: total batch time / ops_per_pe
            double latency_avg    = avg_batch_us / ops_per_pe;
            // aggregate throughput: all PEs × ops_per_pe ops in avg_batch_us
            double throughput_avg = (static_cast<double>(bench.numPes()) * ops_per_pe)
                                    / (avg_batch_us * 1e-6) / 1e6;  // Mops/s
            rows.push_back({ops_per_pe, latency_avg, throughput_avg, size_pass});
        }
    }

    if (!bench.isRoot()) return;

    std::ostringstream js;
    js << std::fixed;
    js << "{\n";
    js << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL") << "\",\n";
    js << "  \"ops_per_pe_unit\": \"calls/PE/round\",\n";
    js << "  \"latency_unit\": \"us\",\n";
    js << "  \"throughput_unit\": \"Mops/s\",\n";
    js << "  \"metrics\": [\n";
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        const auto &r = rows[i];
        js << "    {"
           << "\"ops_per_pe\": "     << r.ops_per_pe     << ", "
           << "\"latency_avg\": "    << r.latency_avg     << ", "
           << "\"throughput_avg\": " << r.throughput_avg  << ", "
           << "\"pass\": "           << (r.pass ? "true" : "false")
           << "}";
        if (i + 1 < static_cast<int>(rows.size())) js << ",";
        js << "\n";
    }
    js << "  ]\n";
    js << "}\n";

    printf("%s", js.str().c_str());
    fflush(stdout);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = {.mpi_comm = &comm};
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int my_pe   = nvshmem_my_pe();
    int num_pes = nvshmem_n_pes();
    CUDA_CHECK(cudaSetDevice(my_pe));

    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, my_pe));

    try {
        NvshmemAtomicBenchmark bench(my_pe, num_pes);
        bench.allocate();
        runTest(bench, clock_khz, kWarmupIters, kBenchIters);
    } catch (const std::exception &e) {
        fprintf(stderr, "Exception on PE %d: %s\n", my_pe, e.what());
        nvshmem_finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    nvshmem_finalize();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
