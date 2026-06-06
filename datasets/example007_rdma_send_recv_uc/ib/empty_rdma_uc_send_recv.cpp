/*
 * Example 7: Two-sided ops – UC SEND/RECV with GPU memory (GPUDirect RDMA)
 *
 * RDMA Unreliable Connection (UC) benchmark using SEND/RECEIVE operations.
 * Data buffers are allocated on the GPU via cudaMalloc and registered directly
 * with libibverbs using GPUDirect RDMA (requires nvidia-peermem kernel module).
 * Two QP endpoints are created on the same IB device, metadata is exchanged
 * over a TCP socket (localhost), and the benchmark sweeps multiple data sizes
 * measuring latency and throughput.
 *
 * Compile:  nvcc -x cu -O2 -std=c++17 ref_rdma_uc_send_recv.cpp \
 *               -o rdma_uc_send_recv -libverbs -pthread -lcudart
 *
 * Output: Exactly ONE JSON object on stdout.  All diagnostics go to stderr.
 */

#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Configuration
// ============================================================================

static constexpr int    IB_PORT         = 1;
static constexpr int    MAX_SEND_WR     = 128;
static constexpr int    MAX_RECV_WR     = 128;
static constexpr int    MAX_SGE         = 1;
static constexpr int    CQ_SIZE         = 256;
static constexpr size_t MAX_BUFFER_SIZE = 4 * 1024 * 1024;  // 4 MB

static int g_gid_index = -1;

// ============================================================================
// Error Handling
// ============================================================================

#define CHECK_ERRNO(cond, msg) \
    do { \
        if (!(cond)) { \
            perror(msg); \
            throw std::runtime_error(std::string(msg) + ": " + strerror(errno)); \
        } \
    } while (0)

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(msg); \
        } \
    } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") \
                                     + cudaGetErrorString(_e)); \
        } \
    } while (0)

// ============================================================================
// QP Connection Metadata
// ============================================================================

struct QpConnInfo {
    uint32_t      qpn;
    uint16_t      lid;
    int           gid_index;
    union ibv_gid gid;
};

// ============================================================================
// TCP Out-of-Band Helpers
// ============================================================================

static int tcpListen(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_ERRNO(fd >= 0, "socket");
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(port);

    CHECK_ERRNO(bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "bind");
    CHECK_ERRNO(listen(fd, 1) == 0, "listen");
    return fd;
}

static int tcpAccept(int listen_fd) {
    int fd = accept(listen_fd, nullptr, nullptr);
    CHECK_ERRNO(fd >= 0, "accept");
    return fd;
}

static int tcpConnect(const char* host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_ERRNO(fd >= 0, "socket");

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, host, &addr.sin_addr);

    CHECK_ERRNO(connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "connect");
    return fd;
}

static void tcpSendAll(int fd, const void* buf, size_t len) {
    auto p = static_cast<const char*>(buf);
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        CHECK_ERRNO(n > 0, "send");
        p   += n;
        len -= static_cast<size_t>(n);
    }
}

static void tcpRecvAll(int fd, void* buf, size_t len) {
    auto p = static_cast<char*>(buf);
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        CHECK_ERRNO(n > 0, "recv");
        p   += n;
        len -= static_cast<size_t>(n);
    }
}

static void exchangeConnInfo(int sock_fd,
                             const QpConnInfo& local,
                             QpConnInfo& remote) {
    tcpSendAll(sock_fd, &local,  sizeof(local));
    tcpRecvAll(sock_fd, &remote, sizeof(remote));
}

// ============================================================================
// RdmaUcBenchmark Class
// ============================================================================

class RdmaUcBenchmark {
public:
    // buffer_ is allocated on GPU via cudaMalloc and registered with
    // libibverbs using GPUDirect RDMA (nvidia-peermem kernel module).
    RdmaUcBenchmark(struct ibv_device* device, size_t buffer_size,
                    struct ibv_context* shared_ctx = nullptr,
                    struct ibv_pd*      shared_pd  = nullptr)
        : device_(device), buffer_size_(buffer_size),
          owns_context_(shared_ctx == nullptr),
          owns_pd_(shared_pd == nullptr)
    {
        if (shared_ctx) {
            ctx_ = shared_ctx;
        } else {
            ctx_ = ibv_open_device(device_);
            CHECK_ERRNO(ctx_, "ibv_open_device");
        }

        gid_index_ = findBestGidIndex();
        int ret = ibv_query_gid(ctx_, IB_PORT, gid_index_, &gid_);
        CHECK(ret == 0, "ibv_query_gid failed");

        struct ibv_port_attr port_attr;
        ret = ibv_query_port(ctx_, IB_PORT, &port_attr);
        CHECK(ret == 0, "ibv_query_port failed");
        lid_ = port_attr.lid;

        if (shared_pd) {
            pd_ = shared_pd;
        } else {
            pd_ = ibv_alloc_pd(ctx_);
            CHECK_ERRNO(pd_, "ibv_alloc_pd");
        }

        cq_ = ibv_create_cq(ctx_, CQ_SIZE, nullptr, nullptr, 0);
        CHECK_ERRNO(cq_, "ibv_create_cq");

        // Allocate GPU memory and register for GPUDirect RDMA
        CUDA_CHECK(cudaMalloc(&buffer_, buffer_size_));
        CUDA_CHECK(cudaMemset(buffer_, 0, buffer_size_));

        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        mr_ = ibv_reg_mr(pd_, buffer_, buffer_size_, access);
        if (!mr_) {
            cudaFree(buffer_);
            buffer_ = nullptr;
            throw std::runtime_error(
                "ibv_reg_mr on GPU memory failed – check nvidia-peermem is loaded");
        }

        createQp();
    }

    ~RdmaUcBenchmark() {
        if (qp_)                     ibv_destroy_qp(qp_);
        if (mr_)                     ibv_dereg_mr(mr_);
        if (buffer_)                 cudaFree(buffer_);
        if (cq_)                     ibv_destroy_cq(cq_);
        if (owns_pd_     && pd_)     ibv_dealloc_pd(pd_);
        if (owns_context_ && ctx_)   ibv_close_device(ctx_);
    }

    RdmaUcBenchmark(const RdmaUcBenchmark&)            = delete;
    RdmaUcBenchmark& operator=(const RdmaUcBenchmark&) = delete;

    // ------------------------------------------------------------------
    // Connection
    // ------------------------------------------------------------------

    QpConnInfo getConnInfo() const {
        QpConnInfo info{};
        info.qpn       = qp_->qp_num;
        info.lid       = lid_;
        info.gid_index = gid_index_;
        info.gid       = gid_;
        return info;
    }

    void connectQp(const QpConnInfo& remote) {
        transitionToInit();
        transitionToRtr(remote);
        transitionToRts();
    }

    // ------------------------------------------------------------------
    // Data-path: SEND / RECV
    // ------------------------------------------------------------------

    bool postSend(size_t offset, size_t size, uint64_t wr_id = 0) {
        // TODO
    }

    bool postRecv(size_t offset, size_t size, uint64_t wr_id = 0) {
        // TODO
    }

    // ------------------------------------------------------------------
    // Completion
    // ------------------------------------------------------------------

    bool pollCompletion(int timeout_ms = 5000) {
        // TODO
    }

    // ------------------------------------------------------------------
    // GPU buffer helpers
    // ------------------------------------------------------------------

    // Copy host data into the GPU buffer.
    // cudaDeviceSynchronize ensures the write is visible to the RDMA NIC
    // (GPUDirect RDMA PCIe DMA) before ibv_post_send is called.
    void copyFromHost(const void* host_src, size_t len, size_t offset = 0) {
        CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer_) + offset,
                              host_src, len, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy GPU buffer data to a host destination.
    // cudaDeviceSynchronize ensures prior RDMA DMA writes into GPU memory are
    // visible to CUDA before we read them back to the host.
    void copyToHost(void* host_dst, size_t len, size_t offset = 0) const {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_dst,
                              static_cast<const char*>(buffer_) + offset,
                              len, cudaMemcpyDeviceToHost));
    }

    // Zero the GPU buffer region and synchronize so the RDMA NIC sees zeros
    // before any incoming write lands.
    void zeroBuffer(size_t len, size_t offset = 0) {
        CUDA_CHECK(cudaMemset(static_cast<char*>(buffer_) + offset, 0, len));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    void*               getBuffer()     const { return buffer_; }
    size_t              getBufferSize() const { return buffer_size_; }
    struct ibv_context* getContext()    const { return ctx_; }
    struct ibv_pd*      getPd()         const { return pd_; }

private:
    // ------------------------------------------------------------------
    // GID selection
    // ------------------------------------------------------------------

    int findBestGidIndex() {
        // TODO
        
    }

    // ------------------------------------------------------------------
    // QP creation (UC)
    // ------------------------------------------------------------------

    void createQp() {
        // TODO
    }

    // ------------------------------------------------------------------
    // QP state transitions
    // ------------------------------------------------------------------

    void transitionToInit() {
        // TODO
    }

    void transitionToRtr(const QpConnInfo& remote) {


        // TODO
    }

    void transitionToRts() {
        // TODO
    }

    // ------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------

    struct ibv_device*  device_;
    struct ibv_context* ctx_        = nullptr;
    struct ibv_pd*      pd_         = nullptr;
    struct ibv_cq*      cq_         = nullptr;
    struct ibv_qp*      qp_         = nullptr;
    struct ibv_mr*      mr_         = nullptr;
    void*               buffer_     = nullptr;   // GPU memory (cudaMalloc)
    size_t              buffer_size_;
    int                 gid_index_;
    union  ibv_gid      gid_;
    uint16_t            lid_;

    bool owns_context_;
    bool owns_pd_;
};

// ============================================================================
// Device Discovery
// ============================================================================

static std::vector<struct ibv_device*> getDeviceList() {
    int num = 0;
    struct ibv_device** list = ibv_get_device_list(&num);
    CHECK_ERRNO(list, "ibv_get_device_list");
    return {list, list + num};
}

static int findUcCapableDevice(const std::vector<struct ibv_device*>& devices) {
    for (int i = 0; i < static_cast<int>(devices.size()); ++i) {
        struct ibv_context* ctx = ibv_open_device(devices[i]);
        if (!ctx) continue;

        struct ibv_pd* pd = ibv_alloc_pd(ctx);
        if (!pd) { ibv_close_device(ctx); continue; }

        struct ibv_cq* cq = ibv_create_cq(ctx, 4, nullptr, nullptr, 0);
        if (!cq) { ibv_dealloc_pd(pd); ibv_close_device(ctx); continue; }

        struct ibv_qp_init_attr qia{};
        qia.send_cq = cq; qia.recv_cq = cq;
        qia.qp_type = IBV_QPT_UC;
        qia.cap.max_send_wr = 1; qia.cap.max_recv_wr = 1;
        qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
        struct ibv_qp* qp = ibv_create_qp(pd, &qia);

        bool ok = (qp != nullptr);
        if (qp) ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);

        if (ok) {
            std::cerr << "Auto-selected UC-capable device [" << i << "] "
                      << ibv_get_device_name(devices[i]) << std::endl;
            return i;
        }
    }
    return -1;
}

// ============================================================================
// Benchmark Driver
// ============================================================================

struct MetricEntry {
    size_t data_size;
    double throughput_avg;   // Gbps
    double latency_avg;      // us
};

static void runTest(RdmaUcBenchmark& sender,
                    RdmaUcBenchmark& receiver,
                    const std::vector<size_t>& data_sizes,
                    int iterations, int warmup)
{
    std::vector<MetricEntry> metrics;
    bool all_correct = true;

    // Staging buffers on the host for initialization and correctness checks.
    // The actual RDMA transfers go directly between GPU buffers.
    std::vector<uint8_t> host_src;
    std::vector<uint8_t> host_dst;

    for (size_t ds : data_sizes) {
        if (ds > sender.getBufferSize() || ds > receiver.getBufferSize()) {
            std::cerr << "Skipping size " << ds
                      << " (exceeds buffer)" << std::endl;
            continue;
        }

        // Build expected pattern on the host and upload to sender GPU buffer
        host_src.resize(ds);
        for (size_t i = 0; i < ds; ++i)
            host_src[i] = static_cast<uint8_t>((i + ds) & 0xFF);
        sender.copyFromHost(host_src.data(), ds);
        receiver.zeroBuffer(ds);

        // Warmup
        for (int w = 0; w < warmup; ++w) {
            CHECK(receiver.postRecv(0, ds, static_cast<uint64_t>(w)),
                  "postRecv warmup failed");
            CHECK(sender.postSend(0, ds, static_cast<uint64_t>(w)),
                  "postSend warmup failed");
            CHECK(sender.pollCompletion(),   "send poll warmup failed");
            CHECK(receiver.pollCompletion(), "recv poll warmup failed");
            receiver.zeroBuffer(ds);
        }

        // Timed iterations
        double total_us = 0.0;
        for (int it = 0; it < iterations; ++it) {
            receiver.zeroBuffer(ds);
            CHECK(receiver.postRecv(0, ds, static_cast<uint64_t>(it)),
                  "postRecv failed");

            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK(sender.postSend(0, ds, static_cast<uint64_t>(it)),
                  "postSend failed");
            CHECK(sender.pollCompletion(),   "send poll failed");
            CHECK(receiver.pollCompletion(), "recv poll failed");
            auto t1 = std::chrono::high_resolution_clock::now();

            total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        // Correctness: copy receiver GPU buffer back to host and compare
        host_dst.resize(ds);
        receiver.copyToHost(host_dst.data(), ds);
        bool correct = (memcmp(host_src.data(), host_dst.data(), ds) == 0);
        if (!correct) {
            std::cerr << "Data mismatch for size " << ds << std::endl;
            all_correct = false;
        }

        double avg_us     = total_us / iterations;
        double throughput = (ds * 8.0 / 1.0e9) / (avg_us / 1.0e6);
        metrics.push_back({ds, throughput, avg_us});
    }

    // Emit JSON to stdout
    std::ostringstream js;
    js << std::fixed;
    js << "{\n"
       << "  \"Correctness\": \""  << (all_correct ? "PASS" : "FAIL") << "\",\n"
       << "  \"data_size_unit\": \"Bytes\",\n"
       << "  \"throughput_unit\": \"Gbps\",\n"
       << "  \"latency_unit\": \"us\",\n"
       << "  \"metrics\": [\n";
    for (size_t i = 0; i < metrics.size(); ++i) {
        js << "    {"
           << "\"data_size\": "      << metrics[i].data_size
           << ", \"throughput_avg\": " << std::setprecision(4) << metrics[i].throughput_avg
           << ", \"latency_avg\": "    << std::setprecision(4) << metrics[i].latency_avg
           << "}";
        if (i + 1 < metrics.size()) js << ",";
        js << "\n";
    }
    js << "  ]\n}";
    std::cout << js.str() << std::endl;
}

// ============================================================================
// Main
// ============================================================================

static void emitFailJson() {
    std::cout << "{\n"
                 "  \"Correctness\": \"FAIL\",\n"
                 "  \"data_size_unit\": \"Bytes\",\n"
                 "  \"throughput_unit\": \"Gbps\",\n"
                 "  \"latency_unit\": \"us\",\n"
                 "  \"metrics\": []\n"
                 "}" << std::endl;
}

int main(int argc, char* argv[]) {
    int nic_idx    = -1;
    int tcp_port   = 19875;
    int iterations = 100;
    int warmup     = 10;

    if (argc > 1) nic_idx     = std::atoi(argv[1]);
    if (argc > 2) tcp_port    = std::atoi(argv[2]);
    if (argc > 3) g_gid_index = std::atoi(argv[3]);
    if (argc > 4) iterations  = std::atoi(argv[4]);
    if (argc > 5) warmup      = std::atoi(argv[5]);

    std::vector<size_t> data_sizes = {
        256, 1024, 4096, 16384, 65536, 262144, 1048576
    };

    // Verify nvidia-peermem is usable by checking a GPU is reachable
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA device found: " << cudaGetErrorString(cuda_err)
                  << std::endl;
        emitFailJson();
        return EXIT_FAILURE;
    }
    std::cerr << "CUDA devices available: " << device_count << std::endl;

    auto devices = getDeviceList();
    if (devices.empty()) {
        std::cerr << "No RDMA devices found" << std::endl;
        emitFailJson();
        return EXIT_FAILURE;
    }

    if (nic_idx < 0) {
        nic_idx = findUcCapableDevice(devices);
        if (nic_idx < 0) {
            std::cerr << "No UC-capable RDMA device found" << std::endl;
            emitFailJson();
            return EXIT_FAILURE;
        }
    } else if (nic_idx >= static_cast<int>(devices.size())) {
        std::cerr << "Invalid NIC index " << nic_idx << std::endl;
        emitFailJson();
        return EXIT_FAILURE;
    }

    try {
        struct ibv_device* dev = devices[nic_idx];
        std::cerr << "IB device : " << ibv_get_device_name(dev) << std::endl;
        std::cerr << "GPU memory (GPUDirect RDMA) : enabled" << std::endl;

        RdmaUcBenchmark sender(dev, MAX_BUFFER_SIZE);
        RdmaUcBenchmark receiver(dev, MAX_BUFFER_SIZE,
                                 sender.getContext(), sender.getPd());

        QpConnInfo sender_info   = sender.getConnInfo();
        QpConnInfo receiver_info = receiver.getConnInfo();

        QpConnInfo sender_remote{}, receiver_remote{};
        int listen_fd = tcpListen(tcp_port);

        std::thread server_thread([&]() {
            int conn = tcpAccept(listen_fd);
            exchangeConnInfo(conn, receiver_info, receiver_remote);
            close(conn);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        int client_fd = tcpConnect("127.0.0.1", tcp_port);
        exchangeConnInfo(client_fd, sender_info, sender_remote);
        close(client_fd);

        server_thread.join();
        close(listen_fd);

        sender.connectQp(sender_remote);
        receiver.connectQp(receiver_remote);
        std::cerr << "QPs connected (UC).  Running benchmark ..." << std::endl;

        runTest(sender, receiver, data_sizes, iterations, warmup);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        emitFailJson();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
