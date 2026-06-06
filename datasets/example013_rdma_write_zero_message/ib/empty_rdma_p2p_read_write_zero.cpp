/*
 * RDMA zero-byte WRITE_WITH_IMM latency benchmark.
 *
 * Measures the latency of sending a zero-byte RDMA WRITE_WITH_IMM between
 * two Queue Pairs on separate NICs. The client posts a zero-byte write
 * carrying a 32-bit immediate value; the server verifies the received
 * immediate data without checking any payload contents.
 * QP metadata is exchanged out-of-band via TCP.
 *
 * Usage:
 *   ./rdma_write_zero server --nic 0 --port 9999
 *   ./rdma_write_zero client --nic 1 --server <host> --port 9999 [--iterations 100]
 */

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#if defined(__linux__)
#include <endian.h>
#else
#define be64toh(x) __builtin_bswap64(x)
#define htobe64(x) __builtin_bswap64(x)
#endif

// ── constants ────────────────────────────────────────────────────────────

static constexpr int PORT_NUM       = 1;
static constexpr int MAX_SEND_WR    = 256;
static constexpr int MAX_RECV_WR    = 256;
static constexpr int MAX_SEND_SGE   = 1;
static constexpr int MAX_RECV_SGE   = 1;
static constexpr int MAX_CQE        = 512;
static constexpr int MIN_RNR_TIMER  = 12;
static constexpr int TIMEOUT        = 14;
static constexpr int RETRY_CNT      = 7;
static constexpr int RNR_RETRY      = 7;
static constexpr int MAX_RD_ATOMIC  = 1;
static constexpr int WARMUP_ITERATIONS   = 5;
static constexpr int DEFAULT_ITERATIONS  = 100;
static constexpr uint32_t EXPECTED_IMM   = 0xDEADBEEF;
static constexpr size_t   MIN_BUF_SIZE   = 4096;

// qpn(4) + gid(16) + addr(8) + rkey(4) + gid_index(4) = 36
static constexpr int META_BYTES = 36;

static int g_gid_index = -1;

#define CHECK_ERRNO(cond, msg) \
    do { if (!(cond)) { perror(msg); throw std::runtime_error(std::string(msg) + ": " + strerror(errno)); } } while(0)
#define CHECK(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(msg); } while(0)

// ── QP metadata ─────────────────────────────────────────────────────────

struct QpMetadata {
    uint32_t qpn;
    union ibv_gid gid;
    uint64_t addr;
    uint32_t rkey;
    int gid_index;
};

static void pack_meta(const QpMetadata& m, char* buf) {
    uint32_t u32;
    u32 = htonl(m.qpn);              memcpy(buf,      &u32, 4);
                                      memcpy(buf + 4,  m.gid.raw, 16);
    uint64_t u64 = htobe64(m.addr);   memcpy(buf + 20, &u64, 8);
    u32 = htonl(m.rkey);             memcpy(buf + 28, &u32, 4);
    int32_t i32 = htonl((int32_t)m.gid_index);
                                      memcpy(buf + 32, &i32, 4);
}

static void unpack_meta(const char* buf, QpMetadata& m) {
    uint32_t u32; uint64_t u64; int32_t i32;
    memcpy(&u32, buf,      4); m.qpn = ntohl(u32);
    memcpy(m.gid.raw, buf + 4, 16);
    memcpy(&u64, buf + 20, 8); m.addr = be64toh(u64);
    memcpy(&u32, buf + 28, 4); m.rkey = ntohl(u32);
    memcpy(&i32, buf + 32, 4); m.gid_index = ntohl(i32);
}

// ── TCP helpers (out-of-band metadata exchange) ─────────────────────────

static int tcp_listen(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_ERRNO(fd >= 0, "socket");
    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    CHECK_ERRNO(bind(fd, (sockaddr*)&addr, sizeof(addr)) == 0, "bind");
    CHECK_ERRNO(listen(fd, 1) == 0, "listen");
    return fd;
}

static int tcp_accept(int listen_fd) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    int conn = accept(listen_fd, (sockaddr*)&peer, &len);
    CHECK_ERRNO(conn >= 0, "accept");
    return conn;
}

static int tcp_connect(const char* host, uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_ERRNO(fd >= 0, "socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    CHECK_ERRNO(inet_pton(AF_INET, host, &addr.sin_addr) == 1, "inet_pton");
    CHECK_ERRNO(connect(fd, (sockaddr*)&addr, sizeof(addr)) == 0, "connect");
    return fd;
}

static void tcp_send_all(int fd, const void* data, size_t len) {
    auto* p = static_cast<const char*>(data);
    while (len) { ssize_t n = send(fd, p, len, 0); CHECK_ERRNO(n > 0, "send"); p += n; len -= n; }
}

static void tcp_recv_all(int fd, void* data, size_t len) {
    auto* p = static_cast<char*>(data);
    while (len) { ssize_t n = recv(fd, p, len, 0); CHECK_ERRNO(n > 0, "recv"); p += n; len -= n; }
}

// ── device enumeration ──────────────────────────────────────────────────

static std::vector<ibv_device*> getDevices() {
    int n = 0;
    ibv_device** list = ibv_get_device_list(&n);
    CHECK_ERRNO(list, "ibv_get_device_list");
    return {list, list + n};
}

// ── RdmaEndpoint (core RDMA functionality, no test/benchmark logic) ─────

class RdmaEndpoint {
public:
    RdmaEndpoint(ibv_device* dev, size_t buf_sz)
        : buf_sz_(buf_sz) {
        ctx_ = ibv_open_device(dev);
        CHECK_ERRNO(ctx_, "ibv_open_device");
        gid_idx_ = pickGidIndex();
        CHECK(ibv_query_gid(ctx_, PORT_NUM, gid_idx_, &gid_) == 0, "ibv_query_gid");
        pd_  = ibv_alloc_pd(ctx_);                             CHECK_ERRNO(pd_, "ibv_alloc_pd");
        cq_  = ibv_create_cq(ctx_, MAX_CQE, nullptr, nullptr, 0); CHECK_ERRNO(cq_, "ibv_create_cq");
        buf_ = aligned_alloc(4096, buf_sz_);                    CHECK_ERRNO(buf_, "aligned_alloc");
        memset(buf_, 0, buf_sz_);
        mr_  = ibv_reg_mr(pd_, buf_, buf_sz_,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        CHECK_ERRNO(mr_, "ibv_reg_mr");
        initQP();
    }

    ~RdmaEndpoint() {
        //TODO
    }

    QpMetadata getMetadata() const {
       //TODO
    }

    void connect(const QpMetadata& remote) {
        // INIT
        //TODO
    }

    bool rdmaWriteImm(uint64_t raddr, uint32_t rkey, uint32_t imm_data) {
        //TODO
    }

    bool postRecv() {
        //TODO
    }

    bool pollCompletion(int timeout_ms = 10000, ibv_wc* out_wc = nullptr) {
        //TODO
    }

private:
    int pickGidIndex() {
        //TODO
    }

    void initQP() {
        //TODO
    }

    ibv_context* ctx_ = nullptr;
    ibv_pd*      pd_  = nullptr;
    ibv_cq*      cq_  = nullptr;
    ibv_qp*      qp_  = nullptr;
    ibv_mr*      mr_  = nullptr;
    void*        buf_ = nullptr;
    size_t       buf_sz_;
    int          gid_idx_;
    union ibv_gid gid_;
};

// ── test runner ─────────────────────────────────────────────────────────

struct TestResult {
    bool correct;
    double latency_avg_us;
};

static TestResult runTest(ibv_device* dev, bool is_server,
                          const char* host, uint16_t port,
                          int warmup_iters, int measure_iters) {
    RdmaEndpoint ep(dev, MIN_BUF_SIZE);
    QpMetadata my = ep.getMetadata();
    int conn;
    QpMetadata remote;

    // TCP-based metadata exchange
    if (is_server) {
        int lfd = tcp_listen(port);
        std::cerr << "server: waiting for client on port " << port << "...\n";
        conn = tcp_accept(lfd);
        close(lfd);
        char buf[META_BYTES];
        pack_meta(my, buf);
        tcp_send_all(conn, buf, META_BYTES);
        tcp_recv_all(conn, buf, META_BYTES);
        unpack_meta(buf, remote);
    } else {
        conn = tcp_connect(host, port);
        char buf[META_BYTES];
        tcp_recv_all(conn, buf, META_BYTES);
        unpack_meta(buf, remote);
        pack_meta(my, buf);
        tcp_send_all(conn, buf, META_BYTES);
    }

    ep.connect(remote);
    int total_iters = warmup_iters + measure_iters;

    if (is_server) {
        // Pre-post all recv WRs
        for (int i = 0; i < total_iters; i++) {
            CHECK(ep.postRecv(), "postRecv");
        }

        // Signal client to start
        tcp_send_all(conn, "GO", 2);

        // Poll all completions and verify imm_data
        bool all_correct = true;
        for (int i = 0; i < total_iters; i++) {
            ibv_wc wc;
            CHECK(ep.pollCompletion(10000, &wc), "poll recv completion");
            if (wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM ||
                ntohl(wc.imm_data) != EXPECTED_IMM) {
                all_correct = false;
            }
        }

        // Send verification result to client
        char result = all_correct ? 'P' : 'F';
        tcp_send_all(conn, &result, 1);
        close(conn);

        std::cerr << "server: verified " << total_iters << " iterations, "
                  << (all_correct ? "PASS" : "FAIL") << "\n";
        return {all_correct, 0.0};

    } else {
        // Wait for server ready signal
        char go[2];
        tcp_recv_all(conn, go, 2);

        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            CHECK(ep.rdmaWriteImm(remote.addr, remote.rkey, EXPECTED_IMM),
                  "rdmaWriteImm warmup");
            CHECK(ep.pollCompletion(), "poll warmup");
        }

        // Measured iterations
        std::vector<double> latencies(measure_iters);
        for (int i = 0; i < measure_iters; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK(ep.rdmaWriteImm(remote.addr, remote.rkey, EXPECTED_IMM),
                  "rdmaWriteImm");
            CHECK(ep.pollCompletion(), "poll send");
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        // Get verification result from server
        char result;
        tcp_recv_all(conn, &result, 1);
        close(conn);

        bool correct = (result == 'P');
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                     / measure_iters;

        std::cerr << "client: " << measure_iters << " iterations, avg latency = "
                  << std::fixed << std::setprecision(2) << avg << " us, "
                  << (correct ? "PASS" : "FAIL") << "\n";
        return {correct, avg};
    }
}

// ── JSON output (exactly one JSON object to stdout) ─────────────────────

static void printResultJson(const TestResult& result, bool is_server) {
    if (is_server) {
        std::cout << "{\n"
                  << "  \"Correctness\": \""
                  << (result.correct ? "PASS" : "FAIL") << "\"\n"
                  << "}" << std::endl;
    } else {
        std::cout << "{\n"
                  << "  \"Correctness\": \""
                  << (result.correct ? "PASS" : "FAIL") << "\",\n"
                  << "  \"latency_unit\": \"us\",\n"
                  << "  \"metrics\": [\n"
                  << "    {\"latency_avg\": " << std::fixed << std::setprecision(2)
                  << result.latency_avg_us << "}\n"
                  << "  ]\n"
                  << "}" << std::endl;
    }
}

// ── main ────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    bool server = false, client = false;
    int nic_idx = 0;
    uint16_t port = 9999;
    std::string host = "127.0.0.1";
    int iterations = DEFAULT_ITERATIONS;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "server")                          server = true;
        else if (a == "client")                          client = true;
        else if (a == "--nic"        && i+1 < argc)      nic_idx = std::atoi(argv[++i]);
        else if (a == "--port"       && i+1 < argc)      port = (uint16_t)std::atoi(argv[++i]);
        else if (a == "--server"     && i+1 < argc)      host = argv[++i];
        else if (a == "--iterations" && i+1 < argc)      iterations = std::atoi(argv[++i]);
        else if (a == "--gid-index"  && i+1 < argc)      g_gid_index = std::atoi(argv[++i]);
    }

    if (!server && !client) {
        std::cerr << "usage: " << argv[0]
                  << " server|client --nic <idx> [--server <host>]"
                     " [--port <n>] [--iterations <n>] [--gid-index <n>]\n";
        return 1;
    }

    auto devs = getDevices();
    if (devs.empty()) { std::cerr << "no RDMA devices\n"; return 1; }
    if (nic_idx >= (int)devs.size()) { std::cerr << "bad NIC index\n"; return 1; }

    std::cerr << (server ? "server" : "client") << ": device="
              << ibv_get_device_name(devs[nic_idx]) << "\n";

    try {
        auto result = runTest(devs[nic_idx], server, host.c_str(), port,
                              WARMUP_ITERATIONS, iterations);
        printResultJson(result, server);
        return result.correct ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
