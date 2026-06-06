/*
 * ref_rdma_p2p_write_shared_receive_queue.cpp
 *
 * RDMA P2P benchmark using Shared Receive Queue (SRQ).
 *
 * Architecture:
 *   SharedReceiveQueue  – wraps ibv_srq, shared across multiple RC QPs so
 *                         that WRITE_WITH_IMM completions from different
 *                         initiator QPs land on one receive queue.
 *   RdmaEndpoint        – one RC QP + 4 KB-aligned CPU buffer + MR.
 *                         RdmaWrite()         – plain write, no CQE on receiver;
 *                                               used for performance benchmarking.
 *                         RdmaWriteWithImm()  – triggers SRQ recv CQE on server;
 *                                               used for per-size correctness check.
 *   runTest()           – drives warmup + timed benchmark + correctness across
 *                         multiple block sizes, coordinated via TCP.
 *
 * Run as TWO SEPARATE PROCESSES (server first):
 *   ./rdma_srq server --nic 0 [--port 9999]
 *   ./rdma_srq client --nic 0 --server 127.0.0.1 [--port 9999]
 *
 * Each process prints EXACTLY ONE JSON object to stdout.
 *   server : {"Correctness":"PASS"} or {"Correctness":"FAIL"}
 *   client : full JSON with correctness + metrics list for every block size
 */

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
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
static constexpr int    kPortNum     = 1;
static constexpr int    kMaxSendWr   = 128;
static constexpr int    kMaxSendSge  = 1;
static constexpr int    kMaxCqe      = 512;
static constexpr int    kMinRnrTimer = 12;
static constexpr int    kTimeout     = 14;
static constexpr int    kRetryCnt    = 7;
static constexpr int    kRnrRetry    = 7;
static constexpr int    kMaxRdAtomic = 1;
// Wire format: qpn(4) + gid(16) + addr(8) + rkey(4) + gid_index(4)
static constexpr int    kMetaBytes   = 36;

static constexpr int    kWarmupIters = 5;
static constexpr int    kBenchIters  = 20;

// Block sizes (MB) exercised in every run.
static const std::vector<size_t> kDataSizesMb = {1, 4, 16, 64, 256};
// Each endpoint allocates this much CPU memory (covers the largest test size).
static constexpr size_t kMaxBufSz = 256UL * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Error helpers
// ─────────────────────────────────────────────────────────────────────────────
static void ThrowErrno(const char* where) {
  throw std::runtime_error(std::string(where) + ": " + std::strerror(errno));
}

#define CHECK_ERRNO(cond, where)              \
  do {                                        \
    if (!(cond)) ThrowErrno(where);           \
  } while (0)

#define CHECK(cond, msg)                      \
  do {                                        \
    if (!(cond)) throw std::runtime_error(msg); \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// TcpChannel – lightweight blocking TCP helper for metadata exchange.
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
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
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
    addr.sin_port   = htons(port);
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
    if (conn_fd_   >= 0) { ::close(conn_fd_);   conn_fd_   = -1; }
    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  }

 private:
  int listen_fd_ = -1;
  int conn_fd_   = -1;
};

// ─────────────────────────────────────────────────────────────────────────────
// QpMetadata + codec – carries the information needed to connect an RC QP.
// ─────────────────────────────────────────────────────────────────────────────
struct QpMetadata {
  uint32_t      qpn;
  union ibv_gid gid;
  uint64_t      addr;
  uint32_t      rkey;
  int           gid_index;
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
    uint32_t u32; uint64_t u64; int32_t i32;
    std::memcpy(&u32, buf,      4);  m->qpn       = ntohl(u32);
    std::memcpy(m->gid.raw, buf + 4, 16);
    std::memcpy(&u64, buf + 20, 8);  m->addr      = be64toh(u64);
    std::memcpy(&u32, buf + 28, 4);  m->rkey      = ntohl(u32);
    std::memcpy(&i32, buf + 32, 4);  m->gid_index = ntohl(i32);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// SharedReceiveQueue – wraps ibv_srq.
// Shared across multiple RC QPs on the server; any WRITE_WITH_IMM targeted
// at any of those QPs consumes one posted receive from this single queue.
// ─────────────────────────────────────────────────────────────────────────────
class SharedReceiveQueue {
 public:
  SharedReceiveQueue(ibv_pd* pd, int max_wr, int max_sge = 1) : pd_(pd) {
    ibv_srq_init_attr attr{};
    attr.attr.max_wr  = max_wr;
    attr.attr.max_sge = max_sge;
    srq_ = ibv_create_srq(pd_, &attr);
    CHECK_ERRNO(srq_, "ibv_create_srq");
  }

  ~SharedReceiveQueue() noexcept { if (srq_) ibv_destroy_srq(srq_); }

  ibv_srq* get() const { return srq_; }

  void PostRecv(void* buf, size_t len, ibv_mr* mr, uint64_t wr_id) {
    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uint64_t>(buf);
    sge.length = static_cast<uint32_t>(len);
    sge.lkey   = mr->lkey;
    ibv_recv_wr wr{};
    wr.wr_id   = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    ibv_recv_wr* bad = nullptr;
    CHECK(ibv_post_srq_recv(srq_, &wr, &bad) == 0, "ibv_post_srq_recv");
  }

 private:
  ibv_pd*  pd_  = nullptr;
  ibv_srq* srq_ = nullptr;
};

// ─────────────────────────────────────────────────────────────────────────────
// RdmaEndpoint – one RC QP + 4 KB-aligned CPU memory MR.
// Core RDMA operations are kept here; no test or benchmark logic.
// ─────────────────────────────────────────────────────────────────────────────
class RdmaEndpoint {
 public:
  RdmaEndpoint(ibv_context* ctx, ibv_pd* pd, ibv_cq* cq,
               size_t buf_sz, SharedReceiveQueue* srq)
      : ctx_(ctx), pd_(pd), cq_(cq), srq_(srq), buf_sz_(buf_sz) {
    // TODO
  }

  ~RdmaEndpoint() noexcept {
    if (qp_) ibv_destroy_qp(qp_);
    if (mr_) ibv_dereg_mr(mr_);
    if (buf_) std::free(buf_);
  }

  ibv_mr* mr() const { return mr_; }

  QpMetadata GetMetadata() const {
   // TODO
  }

  void Connect(const QpMetadata& remote) {
    // RESET → INIT
    // TODO 
  }

  // Plain RDMA_WRITE: silently writes data into remote memory.
  // No CQE is generated on the receiver side; used for benchmarking.
  bool RdmaWrite(uint64_t raddr, uint32_t rkey, size_t len) {
    // TODO 
  }

  // RDMA_WRITE_WITH_IMM: writes data AND generates a recv CQE on the server's
  // SRQ (consuming one posted receive).  Used for correctness checks.
  bool RdmaWriteWithImm(uint64_t raddr, uint32_t rkey,
                        size_t len, uint32_t imm_data) {
     // TODO 
  }

  // Spin-poll the CQ until one send completion arrives.
  bool PollSendCqe() {
    ibv_wc wc{};
    while (ibv_poll_cq(cq_, 1, &wc) == 0) {}
    return wc.status == IBV_WC_SUCCESS;
  }

  template <typename T>
  T* BufAs() const { return static_cast<T*>(buf_); }

 private:
  int PickGidIndex() {
     // TODO 
  }

  void InitQp() {
     // TODO 
  }

  ibv_context*        ctx_     = nullptr;
  ibv_pd*             pd_      = nullptr;
  ibv_cq*             cq_      = nullptr;
  ibv_qp*             qp_      = nullptr;
  ibv_mr*             mr_      = nullptr;
  void*               buf_     = nullptr;
  SharedReceiveQueue* srq_     = nullptr;
  size_t              buf_sz_;
  int                 gid_idx_ = 0;
  union ibv_gid       gid_{};
};

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────
struct BenchMetric {
  size_t data_size_mb;
  double latency_avg_us;
  double throughput_gbps;
};

struct TestResult {
  bool                     correctness{true};
  std::vector<BenchMetric> metrics;
};

// ─────────────────────────────────────────────────────────────────────────────
// runTest – correctness + performance for every block size.
//
// Per block size the protocol is:
//   1. Receive "RDY" from server (server has posted 2 SRQ recvs for this size).
//   2. Fill both local buffers with a deterministic byte pattern.
//   3. Warmup: kWarmupIters plain RdmaWrite() on ep0 (no SRQ recv consumed).
//   4. Benchmark: kBenchIters plain RdmaWrite() on ep0, wall-clock timed.
//   5. Correctness: 1 RdmaWriteWithImm() on each of ep0 and ep1,
//      consuming the 2 SRQ recvs posted by the server.
//   6. Send "OK" to server; receive server's "OK"/"ER" verdict.
// ─────────────────────────────────────────────────────────────────────────────
static TestResult runTest(
    RdmaEndpoint& ep0,
    RdmaEndpoint& ep1,
    const QpMetadata& s0,
    const QpMetadata& s1,
    TcpChannel& ch,
    const std::vector<size_t>& sizes_mb,
    int warmup,
    int iters)
{
  TestResult result;

  for (size_t sz_mb : sizes_mb) {
    const size_t sz = sz_mb * 1024 * 1024;

    // ── 1. Wait for server to be ready for this size ──────────────────────────
    char sig[4]{};
    ch.RecvAll(sig, 3);
    CHECK(std::memcmp(sig, "RDY", 3) == 0, "expected RDY from server");

    // ── 2. Fill local buffers with a deterministic pattern ────────────────────
    auto* b0 = ep0.BufAs<uint8_t>();
    auto* b1 = ep1.BufAs<uint8_t>();
    for (size_t i = 0; i < sz; i++) {
      b0[i] = static_cast<uint8_t>((i + sz_mb) & 0xFF);
      b1[i] = static_cast<uint8_t>((i + sz_mb + 1) & 0xFF);
    }

    // ── 3. Warmup: plain RDMA_WRITE on ep0 (receiver gets no CQE) ────────────
    for (int i = 0; i < warmup; i++) {
      CHECK(ep0.RdmaWrite(s0.addr, s0.rkey, sz), "warmup RdmaWrite");
      CHECK(ep0.PollSendCqe(), "warmup PollSendCqe");
    }

    // ── 4. Timed benchmark: plain RDMA_WRITE on ep0 ───────────────────────────
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
      CHECK(ep0.RdmaWrite(s0.addr, s0.rkey, sz), "bench RdmaWrite");
      CHECK(ep0.PollSendCqe(), "bench PollSendCqe");
    }
    auto t1 = std::chrono::steady_clock::now();

    // ── 5. Correctness: WRITE_WITH_IMM on both QPs (each consumes 1 SRQ recv) ─
    CHECK(ep0.RdmaWriteWithImm(s0.addr, s0.rkey, sz,
                               static_cast<uint32_t>(sz_mb)),
          "correctness RdmaWriteWithImm ep0");
    CHECK(ep0.PollSendCqe(), "correctness PollSendCqe ep0");

    CHECK(ep1.RdmaWriteWithImm(s1.addr, s1.rkey, sz,
                               static_cast<uint32_t>(sz_mb + 100)),
          "correctness RdmaWriteWithImm ep1");
    CHECK(ep1.PollSendCqe(), "correctness PollSendCqe ep1");

    // ── 6. Sync: tell server we are done; wait for its verdict ───────────────
    ch.SendAll("OK", 2);
    char verdict[3]{};
    ch.RecvAll(verdict, 2);
    if (std::memcmp(verdict, "OK", 2) != 0) result.correctness = false;

    // ── Compute metrics ───────────────────────────────────────────────────────
    // throughput (Gbps) = sz_bytes * 8_bits / (avg_us * 1e-6 s) / 1e9
    //                   = sz * 8 / (avg_us * 1000)
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count()
                    / static_cast<double>(iters);
    double gbps   = static_cast<double>(sz) * 8.0 / (avg_us * 1000.0);
    result.metrics.push_back({sz_mb, avg_us, gbps});
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON serialisation
// ─────────────────────────────────────────────────────────────────────────────
static std::string BuildJson(const TestResult& r) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(3);
  o << "{\n";
  o << "  \"Correctness\": \"" << (r.correctness ? "PASS" : "FAIL") << "\",\n";
  o << "  \"data_size_unit\": \"MB\",\n";
  o << "  \"throughput_unit\": \"Gbps\",\n";
  o << "  \"latency_unit\": \"us\",\n";
  o << "  \"metrics\": [\n";
  for (size_t i = 0; i < r.metrics.size(); i++) {
    const auto& m = r.metrics[i];
    o << "    {"
      << "\"data_size\": " << m.data_size_mb << ", "
      << "\"throughput_avg\": " << m.throughput_gbps << ", "
      << "\"latency_avg\": " << m.latency_avg_us
      << "}";
    if (i + 1 < r.metrics.size()) o << ",";
    o << "\n";
  }
  o << "  ]\n}";
  return o.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Server entry point
// ─────────────────────────────────────────────────────────────────────────────
static int RunServer(ibv_device* dev, uint16_t port) {
  TcpChannel ch;
  ch.Listen(port);
  ch.Accept();

  ibv_context* ctx = ibv_open_device(dev);
  CHECK_ERRNO(ctx, "ibv_open_device");
  ibv_pd* pd = ibv_alloc_pd(ctx);
  CHECK_ERRNO(pd, "ibv_alloc_pd");
  ibv_cq* cq = ibv_create_cq(ctx, kMaxCqe, nullptr, nullptr, 0);
  CHECK_ERRNO(cq, "ibv_create_cq");

  // One SRQ shared by ep0 and ep1 – core feature under test.
  SharedReceiveQueue srq(pd, /*max_wr=*/128);

  RdmaEndpoint ep0(ctx, pd, cq, kMaxBufSz, &srq);
  RdmaEndpoint ep1(ctx, pd, cq, kMaxBufSz, &srq);

  // ── Exchange QP metadata ───────────────────────────────────────────────────
  char meta[kMetaBytes];
  QpMetadata m0 = ep0.GetMetadata(), m1 = ep1.GetMetadata();
  QpMetadataCodec::Pack(m0, meta); ch.SendAll(meta, kMetaBytes);
  QpMetadataCodec::Pack(m1, meta); ch.SendAll(meta, kMetaBytes);

  QpMetadata c0{}, c1{};
  ch.RecvAll(meta, kMetaBytes); QpMetadataCodec::Unpack(meta, &c0);
  ch.RecvAll(meta, kMetaBytes); QpMetadataCodec::Unpack(meta, &c1);

  // Signal client that server is ready to connect; then connect QPs.
  ch.SendAll("RDY", 3);
  ep0.Connect(c0);
  ep1.Connect(c1);

  // ── Per-size loop ──────────────────────────────────────────────────────────
  bool all_correct = true;

  for (size_t sz_mb : kDataSizesMb) {
    const size_t sz = sz_mb * 1024 * 1024;

    // Post 2 SRQ recvs: one for the ep0 WRITE_WITH_IMM, one for ep1.
    // Both QPs share the same SRQ – this is the shared-receive-queue feature.
    srq.PostRecv(ep0.BufAs<void>(), sz, ep0.mr(), sz_mb);
    srq.PostRecv(ep1.BufAs<void>(), sz, ep1.mr(), sz_mb + 100);

    // Tell the client we are ready.
    ch.SendAll("RDY", 3);

    // Wait for the client to finish warmup + bench + correctness writes.
    char ok[3]{};
    ch.RecvAll(ok, 2);
    if (std::memcmp(ok, "OK", 2) != 0) {
      all_correct = false;
      ch.SendAll("ER", 2);
      continue;
    }

    // Poll 2 SRQ CQEs (from the two WRITE_WITH_IMM operations).
    bool size_ok = true;
    for (int n = 0; n < 2; n++) {
      ibv_wc wc{};
      while (ibv_poll_cq(cq, 1, &wc) == 0) {}
      if (wc.status  != IBV_WC_SUCCESS ||
          wc.opcode  != IBV_WC_RECV_RDMA_WITH_IMM) {
        size_ok = false;
      }
    }
    if (!size_ok) all_correct = false;
    ch.SendAll(size_ok ? "OK" : "ER", 2);
  }

  std::cout << (all_correct ? "{\"Correctness\":\"PASS\"}"
                            : "{\"Correctness\":\"FAIL\"}") << std::endl;

  ibv_destroy_cq(cq);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  return all_correct ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Client entry point
// ─────────────────────────────────────────────────────────────────────────────
static int RunClient(ibv_device* dev, const std::string& host, uint16_t port) {
  TcpChannel ch;
  ch.Connect(host, port);

  // Receive server QP metadata before creating local resources.
  char meta[kMetaBytes];
  QpMetadata s0{}, s1{};
  ch.RecvAll(meta, kMetaBytes); QpMetadataCodec::Unpack(meta, &s0);
  ch.RecvAll(meta, kMetaBytes); QpMetadataCodec::Unpack(meta, &s1);

  ibv_context* ctx = ibv_open_device(dev);
  CHECK_ERRNO(ctx, "ibv_open_device");
  ibv_pd* pd = ibv_alloc_pd(ctx);
  CHECK_ERRNO(pd, "ibv_alloc_pd");
  ibv_cq* cq = ibv_create_cq(ctx, kMaxCqe, nullptr, nullptr, 0);
  CHECK_ERRNO(cq, "ibv_create_cq");

  // Client never receives; no SRQ needed.
  RdmaEndpoint ep0(ctx, pd, cq, kMaxBufSz, nullptr);
  RdmaEndpoint ep1(ctx, pd, cq, kMaxBufSz, nullptr);

  // Send client metadata and wait for server "RDY".
  char buf[kMetaBytes];
  QpMetadata c0 = ep0.GetMetadata(), c1 = ep1.GetMetadata();
  QpMetadataCodec::Pack(c0, buf); ch.SendAll(buf, kMetaBytes);
  QpMetadataCodec::Pack(c1, buf); ch.SendAll(buf, kMetaBytes);

  char rdy[4]{};
  ch.RecvAll(rdy, 3);
  CHECK(std::memcmp(rdy, "RDY", 3) == 0, "expected initial RDY from server");

  ep0.Connect(s0);
  ep1.Connect(s1);

  // Run correctness verification + performance benchmark.
  TestResult result = runTest(ep0, ep1, s0, s1, ch,
                               kDataSizesMb, kWarmupIters, kBenchIters);

  std::cout << BuildJson(result) << std::endl;

  ibv_destroy_cq(cq);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  return result.correctness ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " server|client --nic <idx> [--server <ip>] [--port <p>]\n";
    return 1;
  }

  const bool  is_server = std::string(argv[1]) == "server";
  const bool  is_client = std::string(argv[1]) == "client";
  int         nic  = 0;
  uint16_t    port = 9999;
  std::string host = "127.0.0.1";

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if      (a == "--nic"    && i + 1 < argc) nic  = std::atoi(argv[++i]);
    else if (a == "--port"   && i + 1 < argc) port = static_cast<uint16_t>(std::atoi(argv[++i]));
    else if (a == "--server" && i + 1 < argc) host = argv[++i];
  }

  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs || nic >= ndev) {
    std::cout << "{\"Correctness\":\"FAIL\"}" << std::endl;
    return 1;
  }

  int rc = 1;
  try {
    if      (is_server) rc = RunServer(devs[nic], port);
    else if (is_client) rc = RunClient(devs[nic], host, port);
    else    std::cerr << "unknown role: " << argv[1] << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    std::cout << "{\"Correctness\":\"FAIL\"}" << std::endl;
  }

  ibv_free_device_list(devs);
  return rc;
}
