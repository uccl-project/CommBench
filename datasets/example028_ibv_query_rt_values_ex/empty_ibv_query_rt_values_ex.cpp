/*
 * Query the NIC runtime clock using ibv_query_rt_values_ex.
    Record the sender-side timestamp before posting the write_with_imm work request.
    Post the RDMA operation and wait for the send completion on the CQ.
    Record another NIC timestamp after the completion is observed.
    Compute the elapsed time in nanoseconds from the raw_clock values.

    Please test the results across different data sizes and compare the latency 
    measured using standard chrono timing and ibv_query_rt_values_ex.
 */

 /*
* Usage
g++ -O2 -std=c++17 ref_ibv_query_rt_values_ex.cpp -libverbs -o ref_ibv_query_rt_values_ex
on server side
./ref_ibv_query_rt_values_ex server --nic 0 --port 9999 --gid-index 1
on client side
./ref_ibv_query_rt_values_ex client --nic 0 --server 127.0.0.1 --port 9999 --gid-index 1
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
static constexpr uint32_t EXPECTED_IMM   = 0xC0FFEE00;
static constexpr size_t   MIN_BUF_SIZE   = 4096;

// qpn(4) + gid(16) + addr(8) + rkey(4) + gid_index(4) + lid(2) = 38
static constexpr int META_BYTES = 38;

static int g_gid_index = -1;

#define CHECK_ERRNO(cond, msg) \
    do { if (!(cond)) { perror(msg); throw std::runtime_error(std::string(msg) + ": " + strerror(errno)); } } while(0)
#define CHECK(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(msg); } while(0)

auto to_ns = [](const timespec& t) {
    // Convert timespec to nanoseconds (uint64_t)
    return (uint64_t)t.tv_sec * 1'000'000'000ULL + (uint64_t)t.tv_nsec;
};

// ── QP metadata ─────────────────────────────────────────────────────────

struct QpMetadata {
    uint32_t qpn;
    union ibv_gid gid;
    uint64_t addr;
    uint32_t rkey;
    int gid_index;
    uint16_t lid;
};

static void pack_meta(const QpMetadata& m, char* buf) {
    uint32_t u32;
    u32 = htonl(m.qpn);              memcpy(buf,      &u32, 4);
                                      memcpy(buf + 4,  m.gid.raw, 16);
    uint64_t u64 = htobe64(m.addr);   memcpy(buf + 20, &u64, 8);
    u32 = htonl(m.rkey);             memcpy(buf + 28, &u32, 4);
    int32_t i32 = htonl((int32_t)m.gid_index);
                                      memcpy(buf + 32, &i32, 4);
    uint16_t u16 = htons(m.lid);     memcpy(buf + 36, &u16, 2);
}

static void unpack_meta(const char* buf, QpMetadata& m) {
    uint32_t u32; uint64_t u64; int32_t i32; uint16_t u16;
    memcpy(&u32, buf,      4); m.qpn = ntohl(u32);
    memcpy(m.gid.raw, buf + 4, 16);
    memcpy(&u64, buf + 20, 8); m.addr = be64toh(u64);
    memcpy(&u32, buf + 28, 4); m.rkey = ntohl(u32);
    memcpy(&i32, buf + 32, 4); m.gid_index = ntohl(i32);
    memcpy(&u16, buf + 36, 2); m.lid = ntohs(u16);
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
        // TODO
    }

    ~RdmaEndpoint() {
        // TODO
    }

    QpMetadata getMetadata() const {
        // TODO
    }

    void connect(const QpMetadata& remote) {
        // TODO
    }

    bool rdmaWriteImm(uint64_t raddr, uint32_t rkey,
                    size_t local_off, size_t len,
                    uint32_t imm_data, ibv_values_ex* ts_start_out = nullptr) {
        // TODO
    }

    bool postRecv() {
        // TODO
    }

    bool pollCompletion(int timeout_ms = 10000, ibv_wc* out_wc = nullptr, ibv_values_ex* ts_end_out = nullptr) {
        // TODO
    }

    void* buf() const { return buf_; }

    template<typename T>
    T* bufAs() const { // TODO
    }
    uint64_t getClockKHz() {
        // TODO
    }
private:
// TODO: more functions can go here if needed
    ibv_context* ctx_ = nullptr;
    ibv_pd*      pd_  = nullptr;
    ibv_cq*      cq_  = nullptr;
    ibv_qp*      qp_  = nullptr;
    ibv_mr*      mr_  = nullptr;
    void*        buf_ = nullptr;
    void*   recv_buf_ = nullptr;
    ibv_mr* recv_mr_  = nullptr;
    size_t       buf_sz_;
    int          gid_idx_;
    union ibv_gid gid_;
    bool         is_roce_ = false;
    uint64_t     hca_core_clock_khz_ = 0;
    uint32_t     raw_clock_comp_mask_ = 0;
};

// DO NOT MODIFY BELOW THIS LINE

// ── test runner ─────────────────────────────────────────────────────────

struct TestResult {
    bool correct;
    double latency_avg_hw_us;
    double latency_avg_chrono_us;
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

        // Fill local buffer with a pattern
        size_t nf = MIN_BUF_SIZE / sizeof(uint32_t);
        uint32_t* p = (uint32_t*)ep.bufAs<uint32_t>();
        for (size_t i = 0; i < nf; i++) p[i] = (uint32_t)i;

        // set up the ibv values_ex struct for timestamp collection
        

        ibv_values_ex ts_start{}, ts_end{};
        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            CHECK(ep.rdmaWriteImm(remote.addr, remote.rkey, 0, MIN_BUF_SIZE, EXPECTED_IMM),
                  "rdmaWriteImm warmup");
            CHECK(ep.pollCompletion(), "poll warmup");
        }

        // Measured iterations
        std::vector<double> hw_latencies(measure_iters);
        std::vector<double> chrono_latencies(measure_iters);
        for (int i = 0; i < measure_iters; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();

            CHECK(ep.rdmaWriteImm(remote.addr, remote.rkey, 0, MIN_BUF_SIZE, EXPECTED_IMM, &ts_start),"rdmaWriteImm");
            CHECK(ep.pollCompletion(10000, nullptr, &ts_end), "poll send");
            
            auto t1 = std::chrono::high_resolution_clock::now();
            uint64_t delta = to_ns(ts_end.raw_clock) - to_ns(ts_start.raw_clock);
            hw_latencies[i] = (double)delta / 1000.0; // raw_clock is timespec in nanoseconds, convert to microseconds
            chrono_latencies[i] = std::chrono::duration<double, std::micro>(t1 - t0).count(); // already in microseconds
        }

        // Get verification result from server
        char result;
        tcp_recv_all(conn, &result, 1);
        close(conn);

        bool correct = (result == 'P');
        double hw_avg = std::accumulate(hw_latencies.begin(), hw_latencies.end(), 0.0)
                     / measure_iters;
        double chrono_avg = std::accumulate(chrono_latencies.begin(), chrono_latencies.end(), 0.0)
                          / measure_iters;

        std::cerr << "client: " << measure_iters << " iterations, hw_avg latency = "
                  << std::fixed << std::setprecision(2) << hw_avg << " us, "
                  << (correct ? "PASS" : "FAIL") << "\n";
        std::cerr << "client: " << measure_iters << " iterations, chrono_avg latency = "
                  << std::fixed << std::setprecision(2) << chrono_avg << " us, "
                  << (correct ? "PASS" : "FAIL") << "\n";
        return {correct, hw_avg, chrono_avg};
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
        double ratio = (result.latency_avg_hw_us > 0.0)
            ? (result.latency_avg_chrono_us - result.latency_avg_hw_us) / result.latency_avg_hw_us
            : 0.0;
        std::cout << "{\n"
                  << "  \"Correctness\": \""
                  << (result.correct ? "PASS" : "FAIL") << "\",\n"
                  << "  \"latency_unit\": \"us\",\n"
                  << "  \"metrics\": [\n"
                  << "    {"
                  << "\"latency_avg\": " << std::fixed << std::setprecision(2) << result.latency_avg_hw_us
                  << ", \"latency_avg_hw_us\": " << result.latency_avg_hw_us
                  << ", \"latency_avg_chrono_us\": " << result.latency_avg_chrono_us
                  << ", \"chrono_overhead_ratio\": " << std::setprecision(4) << ratio
                  << "}\n"
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