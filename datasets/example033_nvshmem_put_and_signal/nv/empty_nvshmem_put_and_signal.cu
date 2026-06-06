/*
 * NVSHMEM Put-and-Signal Benchmark
 *
 * Sweeps over message sizes (8 B to 256 MB) and measures the latency and
 * throughput of the NVSHMEM put+signal communication pattern.  PE 0 acts as
 * the producer and PE 1 acts as the consumer; remaining PEs participate in
 * barriers only.
 *
 * The producer uses multi-block non-blocking puts
 * (nvshmemx_putmem_nbi_block) followed by nvshmem_quiet() and
 * nvshmemx_signal_op().  The consumer waits with
 * nvshmem_uint64_wait_until (CMP_GE) and verifies the received data.
 *
 * Timing is measured on the producer side via CUDA events, capturing the
 * wall time from the first put to after the signal is delivered.
 *
 * Output: Exactly ONE JSON object on stdout.  All diagnostics go to stderr.
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stdio.h>

#include <sstream>
#include <stdexcept>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int    kWarmupIters  = 10;
static constexpr int    kBenchIters   = 50;
static constexpr int    kProducerPe   = 0;
static constexpr int    kConsumerPe   = 1;
static constexpr int    kBlocks       = 64*16;
static constexpr int    kThreads      = 512;
static constexpr size_t kMaxMsgBytes  = 1UL << 28;  // 256 MB

static const size_t kMsgSizes[] = {
    8, 64, 512, 4096, 32768, 262144,
    1UL << 20, 1UL << 22, 1UL << 24, 1UL << 26, 1UL << 27, 1UL << 28
};
static constexpr int kNumSizes =
    static_cast<int>(sizeof(kMsgSizes) / sizeof(kMsgSizes[0]));

// ── CUDA error checking ───────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── Kernels ───────────────────────────────────────────────────────────────────

// Multi-block bulk non-blocking put: each block streams its chunk to consumer.
// Hint: use nvshmemx_putmem_nbi_block per block for the appropriate slice.
__global__ void kernel_producer_put(uint8_t *recv_buf, const uint8_t *send_buf,
                                    size_t msg_bytes, int consumer_pe)
{
    // TODO
}

// Drain all in-flight NBI puts then deliver the signal to consumer.
// Hint: call nvshmem_quiet() then nvshmemx_signal_op(..., NVSHMEM_SIGNAL_SET, ...).
__global__ void kernel_producer_signal(uint64_t *signal, uint64_t sig_val,
                                       int consumer_pe)
{
    // TODO
}

// Consumer spins until signal >= sig_val, then reads one byte to prevent DCE.
// Hint: use nvshmem_uint64_wait_until with NVSHMEM_CMP_GE.
__global__ void kernel_consumer_wait(const uint8_t *recv_buf, uint64_t *signal,
                                     uint64_t sig_val, uint64_t *sink)
{
    // TODO
}

// ── NvshmemPutSignalBenchmark ─────────────────────────────────────────────────

class NvshmemPutSignalBenchmark {
public:
    NvshmemPutSignalBenchmark(int my_pe, int num_pes)
        : my_pe_(my_pe), num_pes_(num_pes) {}

    ~NvshmemPutSignalBenchmark() { freeBuffers(); }

    NvshmemPutSignalBenchmark(const NvshmemPutSignalBenchmark &) = delete;
    NvshmemPutSignalBenchmark &operator=(const NvshmemPutSignalBenchmark &) = delete;

    // Allocate symmetric buffers: send_buf (kMaxMsgBytes), recv_buf (kMaxMsgBytes),
    // signal (uint64_t), sink (uint64_t).  Initialize send_buf to 0xA5, signal to 0.
    void allocate() {
        // TODO
    }

    void freeBuffers() {
        if (send_buf_) { nvshmem_free(send_buf_); send_buf_ = nullptr; }
        if (recv_buf_) { nvshmem_free(recv_buf_); recv_buf_ = nullptr; }
        if (signal_)   { nvshmem_free(signal_);   signal_   = nullptr; }
        if (sink_)     { nvshmem_free(sink_);      sink_     = nullptr; }
    }

    // Reset local signal buffer to 0 (async on stream).
    void resetSignal(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(signal_, 0, sizeof(uint64_t), stream));
    }

    // Launch producer put + signal kernels onto stream.
    // Hint: kernel_producer_put with kBlocks/kThreads, then kernel_producer_signal 1x1.
    void producerSend(size_t msg_bytes, uint64_t sig_val, cudaStream_t stream) {
        // TODO
    }

    // Launch consumer wait kernel onto stream.
    void consumerWait(uint64_t sig_val, cudaStream_t stream) {
        // TODO
    }

    // Verify that recv_buf[0..msg_bytes-1] == 0xA5.
    bool verify(size_t msg_bytes) const {
        // TODO
        return true;
    }

    void barrier()      const { nvshmem_barrier_all(); }
    bool isProducer()   const { return my_pe_ == kProducerPe; }
    bool isConsumer()   const { return my_pe_ == kConsumerPe; }
    bool isRoot()       const { return my_pe_ == 0; }
    int  myPe()         const { return my_pe_; }
    int  numPes()       const { return num_pes_; }

private:
    int       my_pe_;
    int       num_pes_;
    uint8_t  *send_buf_ = nullptr;
    uint8_t  *recv_buf_ = nullptr;
    uint64_t *signal_   = nullptr;
    uint64_t *sink_     = nullptr;
};

// ── runTest ───────────────────────────────────────────────────────────────────

struct MetricEntry {
    size_t data_size;      // bytes
    double latency_avg;    // us
    double throughput_avg; // GB/s
    bool   pass;
};

static void runTest(NvshmemPutSignalBenchmark &bench,
                    int warmup_iters, int bench_iters)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    bool overall_pass = true;
    std::vector<MetricEntry> rows;

    for (int si = 0; si < kNumSizes; ++si) {
        const size_t msg_bytes = kMsgSizes[si];

        // ── Warmup ────────────────────────────────────────────────────────────
        for (int w = 0; w < warmup_iters; ++w) {
            bench.resetSignal(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench.barrier();
            if (bench.isProducer())
                bench.producerSend(msg_bytes, 1, stream);
            else if (bench.isConsumer())
                bench.consumerWait(1, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench.barrier();
        }

        // ── Timed iterations ──────────────────────────────────────────────────
        double total_us = 0.0;
        bool   size_pass = true;

        for (int it = 0; it < bench_iters; ++it) {
            bench.resetSignal(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench.barrier();

            if (bench.isProducer()) {
                CUDA_CHECK(cudaEventRecord(ev_start, stream));
                bench.producerSend(msg_bytes, 1, stream);
                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            } else if (bench.isConsumer()) {
                bench.consumerWait(1, stream);
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench.barrier();

            if (bench.isProducer()) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                total_us += static_cast<double>(ms) * 1e3;
            }

            // Consumer verifies correctness; result is broadcast to all PEs.
            int p = 1;
            if (bench.isConsumer())
                p = bench.verify(msg_bytes) ? 1 : 0;
            MPI_Bcast(&p, 1, MPI_INT, kConsumerPe, MPI_COMM_WORLD);
            size_pass = size_pass && (p != 0);
        }

        overall_pass = overall_pass && size_pass;

        if (bench.isRoot()) {
            double avg_us    = total_us / bench_iters;
            double bw_gbs    = static_cast<double>(msg_bytes)
                               / (avg_us * 1e-6) / 1e9;
            rows.push_back({msg_bytes, avg_us, bw_gbs, size_pass});
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (!bench.isRoot()) return;

    // ── JSON output ───────────────────────────────────────────────────────────
    std::ostringstream js;
    js << std::fixed;
    js << "{\n";
    js << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL") << "\",\n";
    js << "  \"data_size_unit\": \"bytes\",\n";
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

    if (num_pes < 2) {
        if (my_pe == 0) fprintf(stderr, "need at least 2 PEs\n");
        nvshmem_finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        NvshmemPutSignalBenchmark bench(my_pe, num_pes);
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
