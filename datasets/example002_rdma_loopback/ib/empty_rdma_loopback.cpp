/*
 * Description:
 * RDMA loopback benchmark on a single InfiniBand NIC: two QPs on the same
 * device exchange data via RDMA Write. The RdmaEndpoint class has empty
 * methods labeled with `// TODO` — fill them in with ibverbs calls
 * (ibv_open_device, ibv_alloc_pd, ibv_create_cq, ibv_reg_mr, ibv_create_qp,
 * ibv_modify_qp, ibv_post_send/recv, ibv_poll_cq). Keep the class structure
 * and public API unchanged. The test harness (runTest + main) is already
 * wired up — your implementation must make it pass.
 *
 * This example demonstrates RDMA communication using a SINGLE NIC
 * with two queue pairs that communicate with each other (loopback).
 * This works without external network connectivity.
 *
 * The example:
 * 1. Opens one RDMA device
 * 2. Creates two QPs on the same device
 * 3. Connects the QPs to each other
 * 4. Performs RDMA Write operation
 * 5. Verifies data transfer
 * 
 */

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <functional>

static constexpr int PORT_NUM = 1;
static constexpr int MAX_SEND_WR = 16;
static constexpr int MAX_RECV_WR = 16;
static constexpr int MAX_SEND_SGE = 1;
static constexpr int MAX_RECV_SGE = 1;
static constexpr int MAX_CQE = 16;

// QP transition parameters
static constexpr int MIN_RNR_TIMER = 12;
static constexpr int TIMEOUT = 14;
static constexpr int RETRY_CNT = 7;
static constexpr int RNR_RETRY = 7;
static constexpr int MAX_RD_ATOMIC = 1;

// Global GID index (can be overridden by command line or environment)
static int g_gid_index = -1;

// ============================================================================
// Error Handling Macros
// ============================================================================

#define CHECK_ERRNO(cond, msg) \
    do { \
        if (!(cond)) { \
            perror(msg); \
            throw std::runtime_error(std::string(msg) + ": " + strerror(errno)); \
        } \
    } while(0)

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(msg); \
        } \
    } while(0)

// ============================================================================
// Utility Functions
// ============================================================================

void printGid(const union ibv_gid& gid, const char* prefix = "") {
    bool is_ipv4 = (gid.raw[0] == 0 && gid.raw[1] == 0 &&
                    gid.raw[2] == 0 && gid.raw[3] == 0 &&
                    gid.raw[4] == 0 && gid.raw[5] == 0 &&
                    gid.raw[6] == 0 && gid.raw[7] == 0 &&
                    gid.raw[8] == 0 && gid.raw[9] == 0 &&
                    gid.raw[10] == 0xff && gid.raw[11] == 0xff);

    std::cout << prefix;
    if (is_ipv4) {
        std::cout << "IPv4: " << (int)gid.raw[12] << "."
                  << (int)gid.raw[13] << "."
                  << (int)gid.raw[14] << "."
                  << (int)gid.raw[15];
    } else {
        std::cout << "GID: ";
        for (int i = 0; i < 16; i += 2) {
            if (i > 0) std::cout << ":";
            printf("%02x%02x", gid.raw[i], gid.raw[i+1]);
        }
    }
    std::cout << std::endl;
}

std::vector<struct ibv_device*> getDevices() {
    int num_devices = 0;
    struct ibv_device** device_list = ibv_get_device_list(&num_devices);
    CHECK_ERRNO(device_list, "ibv_get_device_list");

    std::vector<struct ibv_device*> devices;
    for (int i = 0; i < num_devices; i++) {
        devices.push_back(device_list[i]);
    }
    return devices;
}

// ============================================================================
// RdmaEndpoint Class - Core RDMA Communication Abstraction
// ============================================================================

/**
 * Metadata for QP connection establishment
 */
struct QpMetadata {
    uint32_t qpn;           // Queue Pair Number
    union ibv_gid gid;      // Global ID
    uint64_t addr;          // Remote buffer address
    uint32_t rkey;          // Remote key for memory access
    int gid_index;          // GID table index
};

/**
 * RdmaEndpoint - Encapsulates RDMA communication resources
 *
 * This class manages:
 * - Device context and protection domain
 * - Completion queue
 * - Queue pair
 * - Memory region and buffer
 *
 * Usage:
 *   RdmaEndpoint ep(device, buffer_size);
 *   ep.connect(remote_metadata);
 *   ep.rdmaWrite(remote_addr, remote_rkey, data, size);
 *   ep.pollCompletion();
 */
class RdmaEndpoint {
public:
    /**
     * Constructor - Initialize RDMA endpoint with device and buffer
     *
     * @param device RDMA device to use
     * @param buffer_size Size of the local buffer to allocate
     * @param shared_ctx Optional shared context (for multiple endpoints on same device)
     * @param shared_pd Optional shared protection domain
     */
    RdmaEndpoint(struct ibv_device* device, size_t buffer_size,
                 struct ibv_context* shared_ctx = nullptr,
                 struct ibv_pd* shared_pd = nullptr)
        : device_(device), buffer_size_(buffer_size),
          owns_context_(shared_ctx == nullptr),
          owns_pd_(shared_pd == nullptr) {

        // Open device context (or use shared)
        if (shared_ctx) {
            ctx_ = shared_ctx;
        } else {
            ctx_ = ibv_open_device(device_);
            CHECK_ERRNO(ctx_, "ibv_open_device");
        }

        // Find best GID index
        gid_index_ = findBestGidIndex();

        // Query GID
        int ret = ibv_query_gid(ctx_, PORT_NUM, gid_index_, &gid_);
        CHECK(ret == 0, "ibv_query_gid failed");

        // Allocate protection domain (or use shared)
        if (shared_pd) {
            pd_ = shared_pd;
        } else {
            pd_ = ibv_alloc_pd(ctx_);
            CHECK_ERRNO(pd_, "ibv_alloc_pd");
        }

        // Create completion queue
        cq_ = ibv_create_cq(ctx_, MAX_CQE, nullptr, nullptr, 0);
        CHECK_ERRNO(cq_, "ibv_create_cq");

        // Allocate and register buffer
        buffer_ = aligned_alloc(4096, buffer_size_);
        CHECK_ERRNO(buffer_, "aligned_alloc");
        memset(buffer_, 0, buffer_size_);

        int access_flags = IBV_ACCESS_LOCAL_WRITE |
                          IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ;
        mr_ = ibv_reg_mr(pd_, buffer_, buffer_size_, access_flags);
        CHECK_ERRNO(mr_, "ibv_reg_mr");

        // Create queue pair
        createQP();
    }

    ~RdmaEndpoint() {
        if (qp_) ibv_destroy_qp(qp_);
        if (mr_) ibv_dereg_mr(mr_);
        if (buffer_) free(buffer_);
        if (cq_) ibv_destroy_cq(cq_);
        if (owns_pd_ && pd_) ibv_dealloc_pd(pd_);
        if (owns_context_ && ctx_) ibv_close_device(ctx_);
    }

    // Prevent copying
    RdmaEndpoint(const RdmaEndpoint&) = delete;
    RdmaEndpoint& operator=(const RdmaEndpoint&) = delete;

    // Allow moving
    RdmaEndpoint(RdmaEndpoint&& other) noexcept = default;
    RdmaEndpoint& operator=(RdmaEndpoint&& other) noexcept = default;

    // ========================================================================
    // Connection Management
    // ========================================================================

    /**
     * Get local endpoint metadata for connection establishment
     */
    QpMetadata getMetadata() const {
        return QpMetadata{
            qp_->qp_num,
            gid_,
            reinterpret_cast<uint64_t>(buffer_),
            mr_->rkey,
            gid_index_
        };
    }

    /**
     * Connect this endpoint to a remote endpoint
     *
     * @param remote Remote endpoint metadata
     */
    void connect(const QpMetadata& remote) {
        transitionToInit();
        transitionToRTR(remote);
        transitionToRTS();
    }

    // ========================================================================
    // RDMA Operations
    // ========================================================================

    /**
     * Perform RDMA Write to remote memory
     *
     * @param remote_addr Remote buffer address
     * @param remote_rkey Remote memory region key
     * @param local_offset Offset into local buffer
     * @param size Number of bytes to write
     * @param wr_id Work request ID for completion tracking
     * @return true if post succeeded
     */
    bool rdmaWrite(uint64_t remote_addr, uint32_t remote_rkey,
                   size_t local_offset, size_t size, uint64_t wr_id = 0) {
        // TODO: build an ibv_sge for buffer_+local_offset, fill an
        // ibv_send_wr with opcode IBV_WR_RDMA_WRITE and IBV_SEND_SIGNALED,
        // set wr.wr.rdma.{remote_addr,rkey}, then ibv_post_send(qp_, ...).
        return false;
    }

    /**
     * Perform RDMA Read from remote memory
     *
     * @param remote_addr Remote buffer address
     * @param remote_rkey Remote memory region key
     * @param local_offset Offset into local buffer
     * @param size Number of bytes to read
     * @param wr_id Work request ID for completion tracking
     * @return true if post succeeded
     */
    bool rdmaRead(uint64_t remote_addr, uint32_t remote_rkey,
                  size_t local_offset, size_t size, uint64_t wr_id = 0) {
        // TODO: same shape as rdmaWrite but with opcode IBV_WR_RDMA_READ.
        return false;
    }

    /**
     * Post a send operation
     *
     * @param local_offset Offset into local buffer
     * @param size Number of bytes to send
     * @param wr_id Work request ID
     * @return true if post succeeded
     */
    bool postSend(size_t local_offset, size_t size, uint64_t wr_id = 0) {
        // TODO: ibv_send_wr with opcode IBV_WR_SEND, IBV_SEND_SIGNALED,
        // sg_list pointing at buffer_+local_offset, then ibv_post_send.
        return false;
    }

    /**
     * Post a receive operation
     *
     * @param local_offset Offset into local buffer
     * @param size Number of bytes to receive
     * @param wr_id Work request ID
     * @return true if post succeeded
     */
    bool postRecv(size_t local_offset, size_t size, uint64_t wr_id = 0) {
        // TODO: ibv_recv_wr with sg_list pointing at buffer_+local_offset,
        // then ibv_post_recv(qp_, ...).
        return false;
    }

    // ========================================================================
    // Completion Handling
    // ========================================================================

    /**
     * Poll for work completion
     *
     * @param timeout_ms Timeout in milliseconds (-1 for infinite)
     * @return true if completion was successful
     */
    bool pollCompletion(int timeout_ms = 5000) {
        // TODO: spin on ibv_poll_cq(cq_, 1, &wc) until n>0 or timeout;
        // return true only when wc.status == IBV_WC_SUCCESS.
        return false;
    }

    /**
     * Poll for multiple completions
     *
     * @param num_completions Number of completions to wait for
     * @param timeout_ms Timeout in milliseconds
     * @return Number of successful completions
     */
    int pollCompletions(int num_completions, int timeout_ms = 5000) {
        // TODO: call pollCompletion num_completions times, returning the
        // count of successes (stop early on the first failure).
        return 0;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    void* getBuffer() const { return buffer_; }
    size_t getBufferSize() const { return buffer_size_; }
    uint32_t getLkey() const { return mr_->lkey; }
    uint32_t getRkey() const { return mr_->rkey; }
    uint32_t getQpNum() const { return qp_->qp_num; }
    int getGidIndex() const { return gid_index_; }
    const union ibv_gid& getGid() const { return gid_; }
    struct ibv_context* getContext() const { return ctx_; }
    struct ibv_pd* getPd() const { return pd_; }

    /**
     * Get buffer as typed pointer
     */
    template<typename T>
    T* getBufferAs() const {
        return static_cast<T*>(buffer_);
    }

private:
    int findBestGidIndex() {
        // TODO: honor RDMA_GID_INDEX env / g_gid_index override; otherwise
        // scan ibv_query_gid entries on PORT_NUM and prefer the first
        // non-zero IPv4-mapped GID, falling back to the first non-zero GID.
        return 0;
    }

    void createQP() {
         // TODO
    }

    void transitionToInit() {
        // TODO
    }

    void transitionToRTR(const QpMetadata& remote) {
         // TODO
    }

    void transitionToRTS() {
        // TODO
    }

    // RDMA resources
    struct ibv_device* device_;
    struct ibv_context* ctx_ = nullptr;
    struct ibv_pd* pd_ = nullptr;
    struct ibv_cq* cq_ = nullptr;
    struct ibv_qp* qp_ = nullptr;
    struct ibv_mr* mr_ = nullptr;
    void* buffer_ = nullptr;
    size_t buffer_size_;
    int gid_index_;
    union ibv_gid gid_;

    // Ownership flags
    bool owns_context_;
    bool owns_pd_;
};

// ============================================================================
// Test harness (correctness + multi-size performance benchmark)
// ============================================================================

struct MetricRow {
    int    data_size_kb     = 0;
    double throughput_gbps  = 0.0;
    double latency_us       = 0.0;
    bool   pass             = false;
};

static std::vector<int> default_sizes_kb() {
    return {64, 256, 1024, 4096, 16384};
}

static std::pair<bool, std::vector<MetricRow>> runTest(
        struct ibv_device* device,
        const std::vector<int>& sizes_kb,
        int warmup_iters,
        int iters) {
    std::vector<MetricRow> rows;
    rows.reserve(sizes_kb.size());
    bool overall_pass = true;

    for (int kb : sizes_kb) {
        const size_t bytes = static_cast<size_t>(kb) * 1024ULL;

        // Two endpoints on the same NIC, sharing context+PD.
        RdmaEndpoint ep1(device, bytes);
        RdmaEndpoint ep2(device, bytes, ep1.getContext(), ep1.getPd());

        QpMetadata meta1 = ep1.getMetadata();
        QpMetadata meta2 = ep2.getMetadata();
        ep1.connect(meta2);
        ep2.connect(meta1);

        // Initialise source data in ep1's buffer.
        float* src_data = ep1.getBufferAs<float>();
        const size_t nf = bytes / sizeof(float);
        for (size_t i = 0; i < nf; i++) src_data[i] = static_cast<float>(i % 1024);

        // Warmup (untimed).
        bool ok = true;
        for (int i = 0; i < warmup_iters && ok; i++) {
            if (!ep1.rdmaWrite(meta2.addr, meta2.rkey, 0, bytes)) { ok = false; break; }
            if (!ep1.pollCompletion()) { ok = false; break; }
        }

        // Measured iterations.
        const int n = std::max(1, iters);
        std::vector<double> times_ms;
        times_ms.reserve(n);
        for (int i = 0; i < n && ok; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (!ep1.rdmaWrite(meta2.addr, meta2.rkey, 0, bytes)) { ok = false; break; }
            if (!ep1.pollCompletion()) { ok = false; break; }
            auto t1 = std::chrono::high_resolution_clock::now();
            times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        // Verify.
        bool verify_ok = ok;
        if (verify_ok) {
            float* dst_data = ep2.getBufferAs<float>();
            for (size_t i = 0; i < nf; i++) {
                if (src_data[i] != dst_data[i]) { verify_ok = false; break; }
            }
        }

        MetricRow row;
        row.data_size_kb = kb;
        row.pass = ok && verify_ok;
        if (!times_ms.empty()) {
            double sum_ms = 0.0;
            for (double t : times_ms) sum_ms += t;
            const double avg_ms = sum_ms / times_ms.size();
            row.latency_us = avg_ms * 1000.0;  // ms -> us
            const double sec = avg_ms / 1000.0;
            row.throughput_gbps = (sec > 0.0)
                ? (static_cast<double>(bytes) * 8.0 / sec / 1e9)
                : 0.0;
        }
        overall_pass = overall_pass && row.pass;
        rows.push_back(row);
    }

    return {overall_pass, rows};
}

static void printJsonResult(bool overall_pass, const std::vector<MetricRow>& rows) {
    std::cout << "{\n";
    std::cout << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL") << "\",\n";
    std::cout << "  \"data_size_unit\": \"KB\",\n";
    std::cout << "  \"throughput_unit\": \"Gbps\",\n";
    std::cout << "  \"latency_unit\": \"us\",\n";
    std::cout << "  \"metrics\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        std::cout << "    {\"data_size\": " << r.data_size_kb
                  << ", \"throughput_avg\": " << r.throughput_gbps
                  << ", \"latency_avg\": " << r.latency_us << "}";
        if (i + 1 != rows.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    int nic_idx = 0;
    int warmup = 5;
    int iters = 20;

    // CLI: <nic_idx> [gid_index] [warmup] [iters] [size_kb ...]
    if (argc > 1) nic_idx = std::atoi(argv[1]);
    if (argc > 2) g_gid_index = std::atoi(argv[2]);
    if (argc > 3) warmup = std::max(0, std::atoi(argv[3]));
    if (argc > 4) iters  = std::max(1, std::atoi(argv[4]));

    std::vector<int> sizes;
    if (argc > 5) {
        for (int i = 5; i < argc; i++) {
            int kb = std::atoi(argv[i]);
            if (kb > 0) sizes.push_back(kb);
        }
    }
    if (sizes.empty()) sizes = default_sizes_kb();

    auto devices = getDevices();
    if (devices.empty()) {
        std::cerr << "No RDMA devices found" << std::endl;
        printJsonResult(false, {});
        return EXIT_FAILURE;
    }
    if (nic_idx >= static_cast<int>(devices.size())) {
        std::cerr << "Invalid NIC index" << std::endl;
        printJsonResult(false, {});
        return EXIT_FAILURE;
    }

    try {
        auto [overall_pass, rows] = runTest(devices[nic_idx], sizes, warmup, iters);
        printJsonResult(overall_pass, rows);
        return overall_pass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        printJsonResult(false, {});
        return EXIT_FAILURE;
    }
}
