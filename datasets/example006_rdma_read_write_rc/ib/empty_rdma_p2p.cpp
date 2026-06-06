/*
 * Description:
 * RDMA point-to-point communication between two NICs over a Reliable
 * Connection (RC) QP. The RdmaEndpoint class has empty methods labeled
 * with `// TODO`. Fill them in using ibverbs (ibv_open_device, ibv_alloc_pd,
 * ibv_create_cq, ibv_reg_mr, ibv_create_qp, ibv_modify_qp, ibv_post_send,
 * ibv_poll_cq, ibv_query_gid, ibv_query_port). Keep the class structure and
 * public API unchanged. The test harness in run_server/run_client/main is
 * already wired up — your implementation must make it pass.
 *
 * Unlike ref_rdma_loopback.cpp (two QPs on the same NIC), this uses
 * separate NICs on each side and exchanges QP metadata over TCP.
 *
 * Usage:
 *   ./rdma_p2p server --nic 0 --port 9999 [--data-size 1024]
 *   ./rdma_p2p client --nic 1 --server <host> --port 9999 [--data-size 1024]
 *
 * On a single machine with two NICs, use 127.0.0.1 or the remote NIC's IP.
 * Across machines, use the server's routable IP.
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
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#if defined(__linux__)
#include <endian.h>
#else
#define be64toh(x) __builtin_bswap64(x)
#define htobe64(x) __builtin_bswap64(x)
#endif

// ── constants (same as loopback ref) ────────────────────────────────────

static constexpr int PORT_NUM       = 1;
static constexpr int MAX_SEND_WR    = 16;
static constexpr int MAX_RECV_WR    = 16;
static constexpr int MAX_SEND_SGE   = 1;
static constexpr int MAX_RECV_SGE   = 1;
static constexpr int MAX_CQE        = 16;
static constexpr int MIN_RNR_TIMER  = 12;
static constexpr int TIMEOUT        = 14;
static constexpr int RETRY_CNT      = 7;
static constexpr int RNR_RETRY      = 7;
static constexpr int MAX_RD_ATOMIC  = 1;

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

// ── RdmaEndpoint (same API as loopback ref, no shared ctx) ──────────────

class RdmaEndpoint {
public:
    RdmaEndpoint(ibv_device* dev, size_t buf_sz)
        : buf_sz_(buf_sz) {
        // TODO: open device, query GID, alloc PD/CQ, allocate and register
        //       the local buffer (IBV_ACCESS_LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ),
        //       then call initQP().
    }

    ~RdmaEndpoint() {
        if (qp_) ibv_destroy_qp(qp_);
        if (mr_) ibv_dereg_mr(mr_);
        if (buf_) free(buf_);
        if (cq_) ibv_destroy_cq(cq_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (ctx_) ibv_close_device(ctx_);
    }

    QpMetadata getMetadata() const {
        return {qp_->qp_num, gid_, reinterpret_cast<uint64_t>(buf_), mr_->rkey, gid_idx_};
    }

    void connect(const QpMetadata& remote) {
        // TODO: drive the QP through INIT -> RTR -> RTS using ibv_modify_qp.
        //       In RTR set is_global=1, hop_limit=255, dgid=remote.gid,
        //       sgid_index=gid_idx_, dest_qp_num=remote.qpn.
    }

    bool rdmaWrite(uint64_t raddr, uint32_t rkey, size_t off, size_t len) {
        // TODO: post a signaled IBV_WR_RDMA_WRITE WR. sg_list addr =
        //       buf_ + off, length = len, lkey = mr_->lkey. Return true on
        //       successful ibv_post_send.
        return false;
    }

    bool pollCompletion(int timeout_ms = 10000) {
        // TODO: busy-poll cq_ with ibv_poll_cq until one completion or
        //       timeout_ms elapses. Return false on WC error / timeout.
        return false;
    }

    void*  buf()     const { return buf_; }
    size_t bufSize() const { return buf_sz_; }
    template<typename T> T* bufAs() const { return static_cast<T*>(buf_); }

private:
    int pickGidIndex() {
        // TODO: pick a valid GID index. Honour g_gid_index and the
        //       RDMA_GID_INDEX env var first; otherwise prefer an
        //       IPv4-mapped GID (raw[10]==0xff && raw[11]==0xff), then any
        //       non-zero entry from the GID table.
        return 0;
    }

    void initQP() {
        // TODO: create an RC QP on (pd_, cq_, cq_) sized with the
        //       MAX_SEND_WR / MAX_RECV_WR / MAX_SEND_SGE / MAX_RECV_SGE
        //       constants above. Store the result in qp_.
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

// ── server / client ─────────────────────────────────────────────────────

static bool run_server(ibv_device* dev, size_t data_size, uint16_t port) {
    std::cout << "server: " << ibv_get_device_name(dev)
              << ", " << (data_size/1024) << " KB, port " << port << "\n";

    RdmaEndpoint ep(dev, data_size);
    QpMetadata my = ep.getMetadata();

    int lfd = tcp_listen(port);
    std::cout << "waiting for client...\n";
    int conn = tcp_accept(lfd);
    close(lfd);

    // exchange metadata
    char buf[META_BYTES];
    pack_meta(my, buf);
    tcp_send_all(conn, buf, META_BYTES);
    tcp_recv_all(conn, buf, META_BYTES);
    QpMetadata remote;
    unpack_meta(buf, remote);
    ep.connect(remote);

    // wait for client to signal completion
    char sig[4];
    tcp_recv_all(conn, sig, 4);
    close(conn);

    // verify
    size_t nf = data_size / sizeof(float);
    float* dst = ep.bufAs<float>();
    for (size_t i = 0; i < nf; i++) {
        if (dst[i] != (float)i) {
            std::cerr << "mismatch at " << i << ": got " << dst[i] << "\n";
            return false;
        }
    }
    std::cout << "PASS\n";
    return true;
}

static bool run_client(ibv_device* dev, size_t data_size,
                       const char* host, uint16_t port) {
    std::cout << "client: " << ibv_get_device_name(dev)
              << " -> " << host << ":" << port << "\n";

    int conn = tcp_connect(host, port);

    char buf[META_BYTES];
    tcp_recv_all(conn, buf, META_BYTES);
    QpMetadata srv;
    unpack_meta(buf, srv);

    RdmaEndpoint ep(dev, data_size);
    QpMetadata my = ep.getMetadata();
    pack_meta(my, buf);
    tcp_send_all(conn, buf, META_BYTES);
    ep.connect(srv);

    // fill source pattern
    size_t nf = data_size / sizeof(float);
    float* src = ep.bufAs<float>();
    for (size_t i = 0; i < nf; i++) src[i] = (float)i;

    // timed RDMA write
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK(ep.rdmaWrite(srv.addr, srv.rkey, 0, data_size), "rdmaWrite");
    CHECK(ep.pollCompletion(), "poll");
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double mb = data_size / (1024.0 * 1024.0);
    double gbps = (mb / 1024.0) / (ms / 1000.0);
    std::cout << "{\"Correctness\": \"PASS\""
              << ", \"throughput_gbps\": " << gbps
              << ", \"latency_ms\": " << ms
              << ", \"data_size_mb\": " << mb << "}\n";

    tcp_send_all(conn, "DONE", 4);
    close(conn);
    std::cout << "PASS\n";
    return true;
}

// ── main ────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    bool server = false, client = false;
    int nic_idx = 0;
    size_t data_kb = 1024;
    uint16_t port = 9999;
    std::string host = "127.0.0.1";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "server")                          server = true;
        else if (a == "client")                          client = true;
        else if (a == "--nic"       && i+1 < argc)       nic_idx = std::atoi(argv[++i]);
        else if (a == "--data-size" && i+1 < argc)       data_kb = std::atol(argv[++i]);
        else if (a == "--port"      && i+1 < argc)       port = (uint16_t)std::atoi(argv[++i]);
        else if (a == "--server"    && i+1 < argc)       host = argv[++i];
        else if (a == "--gid-index" && i+1 < argc)       g_gid_index = std::atoi(argv[++i]);
    }

    if (!server && !client) {
        std::cerr << "usage: " << argv[0] << " server|client --nic <idx> [--server <host>] "
                     "[--port <n>] [--data-size <KB>] [--gid-index <n>]\n";
        return 1;
    }

    size_t data_size = data_kb * 1024;
    auto devs = getDevices();
    if (devs.empty()) { std::cerr << "no RDMA devices\n"; return 1; }
    if (nic_idx >= (int)devs.size()) { std::cerr << "bad NIC index\n"; return 1; }

    try {
        if (server) return run_server(devs[nic_idx], data_size, port) ? 0 : 1;
        else        return run_client(devs[nic_idx], data_size, host.c_str(), port) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
