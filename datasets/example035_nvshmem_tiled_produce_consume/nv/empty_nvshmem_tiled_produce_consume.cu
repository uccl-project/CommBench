/*
 * NVSHMEM Tiled Producer-Consumer Pipeline Benchmark
 *
 * PE 0 (producer) fills a square float matrix and pushes it tile-by-tile to
 * PE 1 (consumer) using nvshmemx::tile_put_block with REMOTE_PUSH_NBI.  After
 * each tile the producer drains the put with nvshmem_quiet and then signals
 * the consumer via nvshmem_int_p.  PE 1 waits on each per-tile flag with
 * nvshmem_int_wait_until and overlaps a partial-sum computation over the
 * received tile with the producer's next put, giving tile-granularity
 * transfer/compute overlap.
 *
 * The benchmark sweeps square matrix sizes (M = N = side) with a fixed tile
 * shape of kTileM x kTileN and reports per-configuration latency (us) and
 * throughput (GB/s).
 *
 * Output: Exactly ONE JSON object on stdout.  All diagnostics go to stderr.
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "device/tile/nvshmemx_tile_api.hpp"
#include <stdint.h>
#include <stdio.h>

#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int kProducerPe  = 0;
static constexpr int kConsumerPe  = 1;
static constexpr int kWarmupIters = 3;
static constexpr int kBenchIters  = 10;
static constexpr int kTileM       = 64;
static constexpr int kTileN       = 64;
static constexpr int kBlockDim    = 256;

// Square matrix side lengths to sweep (all multiples of kTileM=kTileN=64).
// data_size = side * side * sizeof(float) spans 64 KiB ─ 16 MiB.
static constexpr int kSizes[]   = {128, 192, 256, 320, 384, 512, 640, 768,
                                   1024, 1280, 1536, 2048};
static constexpr int kNumSizes  =
    static_cast<int>(sizeof(kSizes) / sizeof(kSizes[0]));
static constexpr int kMaxSize   = kSizes[kNumSizes - 1];
static constexpr int kMaxElems  = kMaxSize * kMaxSize;
static constexpr int kMaxTiles  = (kMaxSize / kTileM) * (kMaxSize / kTileN);

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

// ── Helpers ───────────────────────────────────────────────────────────────────

// Pretty-print a byte count as "N B", "N KiB", or "N.NN MiB".
static std::string human_bytes(size_t b) {
    char buf[32];
    if (b >= (1ULL << 20)) {
        double m = static_cast<double>(b) / (1 << 20);
        if (m == static_cast<double>(static_cast<long>(m)))
            snprintf(buf, sizeof(buf), "%ld MiB", static_cast<long>(m));
        else
            snprintf(buf, sizeof(buf), "%.2f MiB", m);
    } else if (b >= (1ULL << 10)) {
        double k = static_cast<double>(b) / (1 << 10);
        if (k == static_cast<double>(static_cast<long>(k)))
            snprintf(buf, sizeof(buf), "%ld KiB", static_cast<long>(k));
        else
            snprintf(buf, sizeof(buf), "%.2f KiB", k);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", b);
    }
    return std::string(buf);
}

// ── Helper kernels ────────────────────────────────────────────────────────────

// Fill mat[0..n) with val.
__global__ void kernel_fill(float *mat, int n, float val)
{
    // TODO
}

// Count elements of mat[0..n) that differ from expected, atomicAdd to errors.
__global__ void kernel_verify(const float *mat, int n, float expected, int *errors)
{
    // TODO
}

// ── Tensor types (fixed int shape/stride for runtime M/N) ────────────────────

using LayoutT = nvshmemx::Layout<nvshmemx::shape<int, int>,
                                  nvshmemx::stride<int, int>>;
using TensorT = nvshmemx::Tensor<float, LayoutT>;
using CoordT  = nvshmemx::shape<int, int>;

// ── Producer kernel ───────────────────────────────────────────────────────────
//
// For each tile (ti, tj) of the M×N matrix:
//   1. Build source/destination tensors with row-major layout (stride N, 1).
//   2. Call nvshmemx::tile_put_block<..., REMOTE_PUSH_NBI> to push the tile.
//   3. __syncthreads, then thread 0 issues nvshmem_quiet to drain and
//      nvshmem_int_p(flags + tile_idx, tile_idx, consumer_pe) to signal.
__global__ void kernel_producer(float *mat, int *flags, int M, int N,
                                 int consumer_pe)
{
    // TODO
}

// ── Consumer kernel ───────────────────────────────────────────────────────────
//
// For each tile (in ti, tj scan order):
//   1. Thread 0 calls nvshmem_int_wait_until(flags + tile_idx, CMP_EQ, tile_idx).
//   2. __syncthreads, then all threads accumulate tile elements into a sum
//      (simulate overlapped computation; result can be discarded).
__global__ void kernel_consumer(const float *mat, int *flags, int M, int N)
{
    // TODO
}

// ── TiledPipelineBenchmark ────────────────────────────────────────────────────

class TiledPipelineBenchmark {
public:
    TiledPipelineBenchmark(int my_pe, int num_pes)
        : my_pe_(my_pe), num_pes_(num_pes) {}

    ~TiledPipelineBenchmark() { freeBuffers(); }

    TiledPipelineBenchmark(const TiledPipelineBenchmark &) = delete;
    TiledPipelineBenchmark &operator=(const TiledPipelineBenchmark &) = delete;

    // Allocate symmetric buffers: mat (kMaxElems floats), flags (kMaxTiles ints),
    // errors (1 int).  Throw on failure.
    void allocate() {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        // TODO
    }

    void freeBuffers() {
        if (mat_)    { nvshmem_free(mat_);    mat_    = nullptr; }
        if (flags_)  { nvshmem_free(flags_);  flags_  = nullptr; }
        if (errors_) { nvshmem_free(errors_); errors_ = nullptr; }
    }

    // Producer fills its matrix with val; consumer resets its flags to -1
    // (0xFF bytes), since producer signals with tile_idx ≥ 0.
    void prepare(int M, int N, float val) {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        // TODO
    }

    // Launch role-appropriate kernel on one CUDA block of kBlockDim threads;
    // return elapsed wall-clock time in µs via std::chrono.
    double run(int M, int N) {
        CUDA_CHECK(cudaSetDevice(my_pe_));
        auto t0 = std::chrono::high_resolution_clock::now();
        // TODO
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // Consumer runs kernel_verify over its mat_ against expected; producer
    // returns true.  Returns true if no errors are recorded.
    bool verify(int M, int N, float expected) {
        if (my_pe_ != kConsumerPe) return true;
        CUDA_CHECK(cudaSetDevice(my_pe_));
        CUDA_CHECK(cudaMemset(errors_, 0, sizeof(int)));
        // TODO
        int h = 0;
        CUDA_CHECK(cudaMemcpy(&h, errors_, sizeof(int), cudaMemcpyDeviceToHost));
        return h == 0;
    }

    double allreduceMax(double v) const {
        double r = 0.0;
        MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return r;
    }

    bool allreduceAnd(bool v) const {
        int li = v ? 1 : 0, ri = 0;
        MPI_Allreduce(&li, &ri, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return ri != 0;
    }

    void barrier() const { nvshmem_barrier_all(); }

    int  myPe()       const { return my_pe_;  }
    int  numPes()     const { return num_pes_; }
    bool isProducer() const { return my_pe_ == kProducerPe; }

private:
    int    my_pe_;
    int    num_pes_;
    float *mat_    = nullptr;
    int   *flags_  = nullptr;
    int   *errors_ = nullptr;
};

// ── runTest ───────────────────────────────────────────────────────────────────

struct MetricEntry {
    int    side;
    double latency_avg;
    double throughput_avg;
    size_t data_size;
    bool   pass;
};

static void runTest(TiledPipelineBenchmark &bench, int warmup, int bench_iters)
{
    bool overall_pass = true;
    std::vector<MetricEntry> rows;

    for (int si = 0; si < kNumSizes; ++si) {
        int   side = kSizes[si];
        float val  = static_cast<float>(si + 1);

        for (int w = 0; w < warmup; ++w) {
            bench.prepare(side, side, val);
            bench.barrier();
            bench.run(side, side);
            bench.barrier();
        }

        double total_us  = 0.0;
        bool   size_pass = true;
        for (int it = 0; it < bench_iters; ++it) {
            bench.prepare(side, side, val);
            bench.barrier();

            double elapsed = bench.run(side, side);
            double max_us  = bench.allreduceMax(elapsed);
            total_us += max_us;

            bench.barrier();

            bool local_pass = bench.verify(side, side, val);
            bool round_pass = bench.allreduceAnd(local_pass);
            if (!round_pass) size_pass = false;
        }
        overall_pass = overall_pass && size_pass;

        if (bench.isProducer()) {
            double avg_us    = total_us / bench_iters;
            size_t bytes     = static_cast<size_t>(side) * side * sizeof(float);
            double tput_gb   = static_cast<double>(bytes) / (avg_us * 1e-6) / 1e9;
            rows.push_back({side, avg_us, tput_gb, bytes, size_pass});
        }
    }

    if (!bench.isProducer()) return;

    std::ostringstream js;
    js << std::fixed;
    js << "{\n";
    js << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL") << "\",\n";
    js << "  \"tile_M\": " << kTileM << ",\n";
    js << "  \"tile_N\": " << kTileN << ",\n";
    js << "  \"data_size_unit\": \"Bytes\",\n";
    js << "  \"latency_unit\": \"us\",\n";
    js << "  \"throughput_unit\": \"GB/s\",\n";
    js << "  \"metrics\": [\n";
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        const auto &r = rows[i];
        js << "    {"
           << "\"data_size\": "        << r.data_size               << ", "
           << "\"data_size_pretty\": \"" << human_bytes(r.data_size) << "\", "
           << "\"side\": "             << r.side                    << ", "
           << "\"latency_avg\": "      << r.latency_avg             << ", "
           << "\"throughput_avg\": "   << r.throughput_avg          << ", "
           << "\"pass\": "             << (r.pass ? "true" : "false")
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
        fprintf(stderr, "Need at least 2 PEs for producer-consumer benchmark\n");
        nvshmem_finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        TiledPipelineBenchmark bench(my_pe, num_pes);
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
