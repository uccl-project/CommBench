/*
 * ref_rdma_p2p_write_inline.cpp
 *
 * RDMA inline WRITE latency benchmark between two NICs (RC QP).
 *
 * Demonstrates IBV_SEND_INLINE: the HCA copies small payloads directly from
 * the WQE, bypassing DMA from the local MR.  After QP creation the program
 * queries the actual max inline data size supported by the HCA and clamps
 * the payload accordingly.
 *
 * Each process prints EXACTLY ONE JSON object to stdout.
 *  - client: correctness + latency / throughput metrics
 *  - server: correctness only
 */

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <endian.h>
#else
#define be64toh(x) __builtin_bswap64(x)
#define htobe64(x) __builtin_bswap64(x)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kPortNum = 1;
static constexpr int kMaxSendWr = 128;
static constexpr int kMaxRecvWr = 128;
static constexpr int kMaxSendSge = 1;
static constexpr int kMaxRecvSge = 1;
static constexpr int kMaxCqe = 256;

static constexpr int kMinRnrTimer = 12;
static constexpr int kTimeout = 14;
static constexpr int kRetryCnt = 7;
static constexpr int kRnrRetry = 7;
static constexpr int kMaxRdAtomic = 1;

// qpn(4) + gid(16) + addr(8) + rkey(4) + gid_index(4)
static constexpr int kMetaBytes = 36;

// ─────────────────────────────────────────────────────────────────────────────
// Error helpers
// ─────────────────────────────────────────────────────────────────────────────

static void ThrowErrno(const char* where) {
  throw std::runtime_error(std::string(where) + ": " + std::strerror(errno));
}

#define CHECK_ERRNO(cond, where) \
  do {                           \
    if (!(cond)) ThrowErrno(where); \
  } while (0)

#define CHECK(cond, msg)                \
  do {                                  \
    if (!(cond)) throw std::runtime_error(msg); \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// TCP channel
// ─────────────────────────────────────────────────────────────────────────────

class TcpChannel {
 public:
  ~TcpChannel() { Close(); }

  void Listen(uint16_t port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK_ERRNO(listen_fd_ >= 0, "socket");

    int on = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    CHECK_ERRNO(::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) == 0, "bind");
    CHECK_ERRNO(::listen(listen_fd_, 1) == 0, "listen");
  }

  void Accept() {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    conn_fd_ = ::accept(listen_fd_, (sockaddr*)&peer, &len);
    CHECK_ERRNO(conn_fd_ >= 0, "accept");
    ::close(listen_fd_);
    listen_fd_ = -1;
  }

  void Connect(const std::string& host, uint16_t port) {
    conn_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK_ERRNO(conn_fd_ >= 0, "socket");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    CHECK_ERRNO(::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) == 1, "inet_pton");
    CHECK_ERRNO(::connect(conn_fd_, (sockaddr*)&addr, sizeof(addr)) == 0, "connect");
  }

  void SendAll(const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len) {
      ssize_t n = ::send(conn_fd_, p, len, 0);
      CHECK_ERRNO(n > 0, "send");
      p += n;
      len -= static_cast<size_t>(n);
    }
  }

  void RecvAll(void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len) {
      ssize_t n = ::recv(conn_fd_, p, len, 0);
      CHECK_ERRNO(n > 0, "recv");
      p += n;
      len -= static_cast<size_t>(n);
    }
  }

  void Close() {
    if (conn_fd_ >= 0) {
      ::close(conn_fd_);
      conn_fd_ = -1;
    }
    if (listen_fd_ >= 0) {
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
  }

 private:
  int listen_fd_ = -1;
  int conn_fd_ = -1;
};

// ─────────────────────────────────────────────────────────────────────────────
// RDMA metadata
// ─────────────────────────────────────────────────────────────────────────────

struct QpMetadata {
  uint32_t qpn;
  union ibv_gid gid;
  uint64_t addr;
  uint32_t rkey;
  int gid_index;
};

class QpMetadataCodec {
 public:
  static void Pack(const QpMetadata& m, char* buf) {
    uint32_t u32 = htonl(m.qpn);
    std::memcpy(buf, &u32, 4);
    std::memcpy(buf + 4, m.gid.raw, 16);
    uint64_t u64 = htobe64(m.addr);
    std::memcpy(buf + 20, &u64, 8);
    u32 = htonl(m.rkey);
    std::memcpy(buf + 28, &u32, 4);
    int32_t i32 = htonl(m.gid_index);
    std::memcpy(buf + 32, &i32, 4);
  }

  static void Unpack(const char* buf, QpMetadata* m) {
    uint32_t u32;
    uint64_t u64;
    int32_t i32;
    std::memcpy(&u32, buf, 4);
    m->qpn = ntohl(u32);
    std::memcpy(m->gid.raw, buf + 4, 16);
    std::memcpy(&u64, buf + 20, 8);
    m->addr = be64toh(u64);
    std::memcpy(&u32, buf + 28, 4);
    m->rkey = ntohl(u32);
    std::memcpy(&i32, buf + 32, 4);
    m->gid_index = ntohl(i32);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// RDMA endpoint
// ─────────────────────────────────────────────────────────────────────────────

class RdmaEndpoint {
 public:
  RdmaEndpoint(ibv_device* dev, size_t buf_sz)
      : buf_sz_(buf_sz) {
    ctx_ = ibv_open_device(dev);
    CHECK_ERRNO(ctx_, "ibv_open_device");

    gid_idx_ = PickGidIndex();
    CHECK(ibv_query_gid(ctx_, kPortNum, gid_idx_, &gid_) == 0, "ibv_query_gid");

    pd_ = ibv_alloc_pd(ctx_);
    CHECK_ERRNO(pd_, "ibv_alloc_pd");

    cq_ = ibv_create_cq(ctx_, kMaxCqe, nullptr, nullptr, 0);
    CHECK_ERRNO(cq_, "ibv_create_cq");

    buf_ = std::aligned_alloc(4096, buf_sz_);
    CHECK_ERRNO(buf_, "aligned_alloc");
    std::memset(buf_, 0, buf_sz_);

    mr_ = ibv_reg_mr(pd_, buf_, buf_sz_,
                     IBV_ACCESS_LOCAL_WRITE |
                     IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ);
    CHECK_ERRNO(mr_, "ibv_reg_mr");

    InitQp();
  }

  ~RdmaEndpoint() {
    if (qp_) ibv_destroy_qp(qp_);
    if (mr_) ibv_dereg_mr(mr_);
    if (buf_) std::free(buf_);
    if (cq_) ibv_destroy_cq(cq_);
    if (pd_) ibv_dealloc_pd(pd_);
    if (ctx_) ibv_close_device(ctx_);
  }

  QpMetadata GetMetadata() const {
    return {qp_->qp_num, gid_, reinterpret_cast<uint64_t>(buf_), mr_->rkey, gid_idx_};
  }

  void Connect(const QpMetadata& remote) {
    // TODO: transition QP to RTR and then RTS using the remote metadata
  }

  uint32_t MaxInlineSize() const { return max_inline_size_; }

  bool RdmaWrite(uint64_t raddr, uint32_t rkey, size_t len) {
    i// TODO: post an RDMA WRITE with the given remote address, rkey, and length (payload is dummy, can be on stack)
  }

  bool Poll() {
    //// TODO
  }

  template <typename T>
  T* BufAs() const {
    return static_cast<T*>(buf_);
  }

 private:
  int PickGidIndex() {
    // TODO: pick a GID index with non-zero GID, prefer one that looks like IPv4-mapped (ff:ff in bytes 10-11)
  }

  void InitQp() {
   // TODO: populate max_inline_size_ by querying the QP attributes after creation; transition QP to INIT state
  }

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_qp* qp_ = nullptr;
  ibv_mr* mr_ = nullptr;
  void* buf_ = nullptr;

  size_t buf_sz_;
  int gid_idx_;
  uint32_t max_inline_size_ = 0;
  union ibv_gid gid_{};
};

// ─────────────────────────────────────────────────────────────────────────────
// Entry points
// ─────────────────────────────────────────────────────────────────────────────

static int RunServer(ibv_device* dev, size_t buf_size, uint16_t port) {
  TcpChannel ch;
  ch.Listen(port);
  ch.Accept();

  RdmaEndpoint ep(dev, buf_size);
  QpMetadata my = ep.GetMetadata();

  char meta[kMetaBytes];
  QpMetadataCodec::Pack(my, meta);
  ch.SendAll(meta, kMetaBytes);

  ch.RecvAll(meta, kMetaBytes);
  QpMetadata remote{};
  QpMetadataCodec::Unpack(meta, &remote);
  ep.Connect(remote);

  // Receive actual inline data size from client
  uint32_t actual_size = 0;
  ch.RecvAll(&actual_size, sizeof(actual_size));

  // Wait for client to signal that RDMA WRITE is complete
  char sig[4];
  ch.RecvAll(sig, 4);

  size_t nf = actual_size / sizeof(float);
  float* dst = ep.BufAs<float>();
  bool correct = true;
  for (size_t i = 0; i < nf; i++) {
    if (dst[i] != static_cast<float>(i)) {
      correct = false;
      break;
    }
  }

  // Send verification result back to client
  char result = correct ? 'P' : 'F';
  ch.SendAll(&result, 1);

  std::cerr << "server: max_inline=" << ep.MaxInlineSize()
            << " actual_data=" << actual_size << " bytes\n";
  std::cout << "{\"Correctness\":\"" << (correct ? "PASS" : "FAIL") << "\"}";
  return correct ? 0 : 1;
}

static int RunClient(ibv_device* dev, size_t data_size,
                     const std::string& host, uint16_t port) {
  TcpChannel ch;
  ch.Connect(host, port);

  char meta[kMetaBytes];
  ch.RecvAll(meta, kMetaBytes);
  QpMetadata srv{};
  QpMetadataCodec::Unpack(meta, &srv);

  // Allocate buffer large enough (at least 4096 for MR registration)
  size_t buf_size = std::max(data_size, static_cast<size_t>(4096));
  RdmaEndpoint ep(dev, buf_size);
  QpMetadata my = ep.GetMetadata();
  QpMetadataCodec::Pack(my, meta);
  ch.SendAll(meta, kMetaBytes);
  ep.Connect(srv);

  // Clamp data size to actual HCA inline limit
  uint32_t max_inline = ep.MaxInlineSize();
  uint32_t actual_size = static_cast<uint32_t>(
      std::min(data_size, static_cast<size_t>(max_inline)));
  // Align down to sizeof(float) for clean correctness check
  actual_size = (actual_size / sizeof(float)) * sizeof(float);
  CHECK(actual_size > 0, "inline size too small for even one float");

  std::cerr << "client: max_inline=" << max_inline
            << " actual_data=" << actual_size << " bytes\n";

  // Send actual data size to server so it knows how many bytes to verify
  ch.SendAll(&actual_size, sizeof(actual_size));

  size_t nf = actual_size / sizeof(float);
  float* src = ep.BufAs<float>();
  for (size_t i = 0; i < nf; i++) src[i] = static_cast<float>(i);

  auto t0 = std::chrono::high_resolution_clock::now();
  CHECK(ep.RdmaWrite(srv.addr, srv.rkey, actual_size), "rdmaWrite inline");
  CHECK(ep.Poll(), "poll");
  auto t1 = std::chrono::high_resolution_clock::now();

  // Signal server that RDMA WRITE is complete
  ch.SendAll("DONE", 4);

  // Get verification result from server
  char result;
  ch.RecvAll(&result, 1);
  bool correct = (result == 'P');

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  double mbps = (actual_size * 8.0) / us;  // Mbps (small payload)

  std::cout << "{"
            << "\"Correctness\":\"" << (correct ? "PASS" : "FAIL") << "\","
            << "\"data_size_unit\":\"bytes\","
            << "\"throughput_unit\":\"Mbps\","
            << "\"latency_unit\":\"us\","
            << "\"max_inline_size\":" << max_inline << ","
            << "\"metrics\":[{"
            << "\"data_size\":" << actual_size << ","
            << "\"throughput_avg\":" << mbps << ","
            << "\"latency_avg\":" << us
            << "}]"
            << "}";

  return correct ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " server|client --nic <idx> [--server <ip>] [--port <p>]"
                 " [--data-size <bytes>]\n";
    return 1;
  }

  bool is_server = std::string(argv[1]) == "server";
  bool is_client = std::string(argv[1]) == "client";

  int nic = 0;
  size_t data_size = 256;  // bytes (must fit within HCA inline limit)
  uint16_t port = 9999;
  std::string host = "127.0.0.1";

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--nic" && i + 1 < argc) nic = std::atoi(argv[++i]);
    else if (a == "--data-size" && i + 1 < argc) data_size = std::atol(argv[++i]);
    else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (a == "--server" && i + 1 < argc) host = argv[++i];
  }

  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs || nic >= ndev) return 1;

  // Buffer must be large enough for MR registration (at least 4096)
  size_t buf_size = std::max(data_size, static_cast<size_t>(4096));

  try {
    if (is_server) return RunServer(devs[nic], buf_size, port);
    if (is_client) return RunClient(devs[nic], data_size, host, port);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    std::cout << "{\"Correctness\":\"FAIL\"}";
    return 1;
  }

  return 0;
}
