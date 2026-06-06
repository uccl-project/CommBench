/*
 * NVSHMEM Broadcast Benchmark
 *
 * The root PE (PE 0) broadcasts a data buffer to all PEs each round via
 * nvshmem_broadcastmem.  nvshmem_barrier_all is called after each broadcast
 * to ensure all PEs have received the data before proceeding to the next
 * round.  Correctness is verified every round — each PE checks that its
 * received buffer matches the expected broadcast value.
 *
 * The benchmark sweeps over several buffer sizes and reports per-size
 * latency (us) and throughput (GB/s).
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
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int    kRootPe      = 0;
static constexpr int    kWarmupIters = 5;
static constexpr int    kBenchIters  = 50;
static constexpr size_t kSizes[]     = {256, 1024, 4096, 16384, 65536,
                                        262144, 1048576, 4194304};
static constexpr int    kNumSizes    =
    static_cast<int>(sizeof(kSizes) / sizeof(kSizes[0]));
static constexpr size_t kMaxSize     = kSizes[kNumSizes - 1];

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

// ── Kernel: fill send buffer with a known pattern ─────────────────────────────

__global__ void kernel_fill(uint8_t *buf, size_t n, uint8_t val)
{
    // TODO
}

// ── Kernel: verify receive buffer matches expected value ──────────────────────

__global__ void kernel_verify(const uint8_t *buf, size_t n,
                               uint8_t expected, int *errors)
{
    // TODO
}

// ── NvshmemBroadcastBenchmark ─────────────────────────────────────────────────

class NvshmemBroadcastBenchmark {
public:
    NvshmemBroadcastBenchmark(int my_pe, int num_pes)
        : my_pe_(my_pe), num_pes_(num_pes) {}

    ~NvshmemBroadcastBenchmark() { freeBuffers(); }

    NvshmemBroadcastBenchmark(const NvshmemBroadcastBenchmark &) = delete;
    NvshmemBroadcastBenchmark &operator=(const NvshmemBroadcastBenchmark &) = delete;

    void allocate() {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        // TODO
    }

    void freeBuffers() {
        if (buf_)    { nvshmem_free(buf_);    buf_    = nullptr; }
        if (errors_) { nvshmem_free(errors_); errors_ = nullptr; }
    }

    // Root PE fills its buffer with val; all PEs clear their error counter.
    void prepare(size_t size, uint8_t val) {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        CUDA_CHECK(cudaMemset(errors_, 0, sizeof(int)));
        if (my_pe_ == kRootPe) {
            int threads = 256;
            int blocks  = static_cast<int>((size + threads - 1) / threads);
            kernel_fill<<<blocks, threads>>>(buf_, size, val);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Root broadcasts buf_[0..size) to all PEs; returns elapsed time in us.
    double broadcast(size_t size) {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        auto t0 = std::chrono::high_resolution_clock::now();
        // TODO
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // Each PE verifies that its buffer matches expected val.
    bool verify(size_t size, uint8_t expected) {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        CUDA_CHECK(cudaMemset(errors_, 0, sizeof(int)));
        // TODO
        int h_errors = 0;
        CUDA_CHECK(cudaMemcpy(&h_errors, errors_, sizeof(int), cudaMemcpyDeviceToHost));
        return h_errors == 0;
    }

    // MPI allreduce max of a per-PE double.
    double allreduceMax(double local_val) const {
        double result = 0.0;
        MPI_Allreduce(&local_val, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return result;
    }

    // MPI allreduce AND of a per-PE bool.
    bool allreduceAnd(bool local_val) const {
        int local_int = local_val ? 1 : 0, result_int = 0;
        MPI_Allreduce(&local_int, &result_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return result_int != 0;
    }

    void barrier() const { nvshmem_barrier_all(); }

    int  myPe()   const { return my_pe_;      }
    int  numPes() const { return num_pes_;     }
    bool isRoot() const { return my_pe_ == kRootPe; }

private:
    int      my_pe_;
    int      num_pes_;
    uint8_t *buf_    = nullptr;
    int     *errors_ = nullptr;
};

// ── runTest ───────────────────────────────────────────────────────────────────

struct MetricEntry {
    size_t data_size;
    double latency_avg;
    double throughput_avg;
    bool   pass;
};

static void runTest(NvshmemBroadcastBenchmark &bench,
                    int warmup_iters, int bench_iters)
{
    bool overall_pass = true;
    std::vector<MetricEntry> rows;

    for (int si = 0; si < kNumSizes; ++si) {
        const size_t  size = kSizes[si];
        const uint8_t val  = static_cast<uint8_t>((si + 1) & 0xFF);

        for (int w = 0; w < warmup_iters; ++w) {
            bench.prepare(size, val);
            bench.barrier();
            bench.broadcast(size);
            bench.barrier();
        }

        double total_us  = 0.0;
        bool   size_pass = true;
        for (int it = 0; it < bench_iters; ++it) {
            bench.prepare(size, val);
            bench.barrier();

            double elapsed_us = bench.broadcast(size);
            double max_us     = bench.allreduceMax(elapsed_us);
            total_us += max_us;

            bench.barrier();

            bool local_pass = bench.verify(size, val);
            bool round_pass = bench.allreduceAnd(local_pass);
            if (!round_pass) size_pass = false;
        }
        overall_pass = overall_pass && size_pass;

        if (bench.isRoot()) {
            double avg_us        = total_us / bench_iters;
            double bytes_total   = static_cast<double>(size) * (bench.numPes() - 1);
            double throughput_gb = bytes_total / (avg_us * 1e-6) / 1e9;
            rows.push_back({size, avg_us, throughput_gb, size_pass});
        }
    }

    if (!bench.isRoot()) return;

    std::ostringstream js;
    js << std::fixed;
    js << "{\n";
    js << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL") << "\",\n";
    js << "  \"data_size_unit\": \"Bytes\",\n";
    js << "  \"latency_unit\": \"us\",\n";
    js << "  \"throughput_unit\": \"GB/s\",\n";
    js << "  \"metrics\": [\n";
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        const auto &r = rows[i];
        js << "    {"
           << "\"data_size\": "      << r.data_size      << ", "
           << "\"latency_avg\": "    << r.latency_avg     << ", "
           << "\"throughput_avg\": " << r.throughput_avg  << ", "
           << "\"pass\": "           << (r.pass ? "true" : "false")
           << "}";
        if (i + 1 < static_cast<int>(rows.size())) js << ",";
        js << "\n";
    }
    js << "  ]\n}\n";

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

    if (num_pes < 2) {
        fprintf(stderr, "Need at least 2 PEs for broadcast benchmark\n");
        nvshmem_finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        NvshmemBroadcastBenchmark bench(my_pe, num_pes);
        bench.allocate();
        runTest(bench, kWarmupIters, kBenchIters);
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
