/*
 * ref_rdma_write_scatter_gather_lists.cpp
 *
 * RDMA WRITE using a Scatter-Gather List (SGL) with multiple SGEs.
 *
 * The client allocates kNumSge independent GPU buffers (via hipMalloc),
 * registers each as its own Memory Region (each with its own addr/len/lkey),
 * and issues a single ibv_post_send carrying a kNumSge-entry SGE list that
 * gathers all non-contiguous GPU buffers in one RDMA WRITE to the server's
 * contiguous GPU buffer.
 *
 * This is designed to be run as TWO SEPARATE PROCESSES:
 *   - one server process
 *   - one client process
 *
 * The server and client exchange QP metadata over TCP, then the client
 * benchmarks RDMA WRITE SGL to the server. The server verifies correctness.
 *
 * Each process prints EXACTLY ONE JSON object to stdout.
 *  - client: correctness + performance metrics
 *  - server: correctness only
 */

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define gpuSuccess        hipSuccess
#define gpuError_t        hipError_t
#define gpuGetErrorString hipGetErrorString
// Buffer allocation strategy:
//   ibv_reg_mr requires memory whose physical pages can be pinned by the
//   kernel RDMA driver via pin_user_pages().  Memory allocated through
//   hipMalloc (GPU VRAM) or hipHostMalloc (ROCm-managed pages) sits in
//   allocator-private page tables that pin_user_pages() cannot reach on
//   systems without GPU Direct RDMA kernel support (e.g. ionic NICs, APU
//   unified-memory systems such as MI300X).  Standard heap memory allocated
//   with aligned_alloc lives in the normal CPU page table and is pinnable
//   by any ibverbs driver.  On platforms where GPU Direct RDMA is confirmed
//   working, swap the AllocXxxBuffer helpers below to use hipMalloc /
//   hipFree instead.
#else
#error "This file requires HIP. Define __HIP_PLATFORM_AMD__, __HIP_PLATFORM_NVIDIA__, or __HIPCC__."
#endif

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

static constexpr int kPortNum    = 1;
static constexpr int kNumSge     = 2;        // number of non-contiguous GPU SGE buffers
static constexpr int kMaxSendWr  = 128;
static constexpr int kMaxRecvWr  = 128;
static constexpr int kMaxSendSge = kNumSge;  // QP must support kNumSge SGEs per WR
static constexpr int kMaxRecvSge = 1;
static constexpr int kMaxCqe     = 256;

// Series of data sizes (in MB) to benchmark.
static const size_t kTestSizesMB[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
static constexpr int kNumTestSizes = sizeof(kTestSizesMB) / sizeof(kTestSizesMB[0]);

static constexpr int kMinRnrTimer = 12;
static constexpr int kTimeout     = 14;
static constexpr int kRetryCnt    = 7;
static constexpr int kRnrRetry    = 7;
static constexpr int kMaxRdAtomic = 1;

// Wire format: qpn(4) + gid(16) + addr(8) + rkey(4) + gid_index(4) = 36 bytes
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
      p   += n;
      len -= static_cast<size_t>(n);
    }
  }

  void RecvAll(void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len) {
      ssize_t n = ::recv(conn_fd_, p, len, 0);
      if (n == 0) throw std::runtime_error("recv: connection closed by peer");
      CHECK_ERRNO(n > 0, "recv");
      p   += n;
      len -= static_cast<size_t>(n);
    }
  }

  void Close() {
    if (conn_fd_   >= 0) { ::close(conn_fd_);   conn_fd_   = -1; }
    if (listen_fd_ >= 0) { ::close(listen_fd_);  listen_fd_ = -1; }
  }

 private:
  int listen_fd_ = -1;
  int conn_fd_   = -1;
};

// ─────────────────────────────────────────────────────────────────────────────
// RDMA metadata
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
    std::memcpy(buf,      &u32,      4);
    std::memcpy(buf + 4,  m.gid.raw, 16);
    uint64_t u64 = htobe64(m.addr);
    std::memcpy(buf + 20, &u64,      8);
    u32 = htonl(m.rkey);
    std::memcpy(buf + 28, &u32,      4);
    int32_t i32 = htonl(m.gid_index);
    std::memcpy(buf + 32, &i32,      4);
  }

  static void Unpack(const char* buf, QpMetadata* m) {
    uint32_t u32; uint64_t u64; int32_t i32;
    std::memcpy(&u32, buf,      4); m->qpn = ntohl(u32);
    std::memcpy(m->gid.raw, buf + 4, 16);
    std::memcpy(&u64, buf + 20, 8); m->addr = be64toh(u64);
    std::memcpy(&u32, buf + 28, 4); m->rkey = ntohl(u32);
    std::memcpy(&i32, buf + 32, 4); m->gid_index = ntohl(i32);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// RDMA endpoint – GPU memory backed, SGL-aware
// ─────────────────────────────────────────────────────────────────────────────

class RdmaEndpoint {
 public:
  // buf_sz: total data size (must be divisible by kNumSge * sizeof(float)).
  //   is_server=true  → one contiguous GPU buffer registered as one MR.
  //   is_server=false → kNumSge independent GPU buffers, each its own MR,
  //                     with distinct addr/len/lkey (non-contiguous SGEs).
  RdmaEndpoint(ibv_device* dev, size_t buf_sz, bool is_server)
      : buf_sz_(buf_sz), is_server_(is_server) {
   // TODO: implement this constructor to set up the RDMA endpoint according to the
  }

  ~RdmaEndpoint() {
    // TODO: 
  }

  // Returns the metadata for the server's contiguous remote GPU buffer.
  QpMetadata GetMetadata() const {
    // TODO: 
  }

  void Connect(const QpMetadata& remote) {
    // TODO: 
  }

  // Client: post one RDMA WRITE WR with kNumSge SGEs.
  // Each SGE[i] carries a distinct GPU buffer address, length, and lkey,
  // implementing a true Scatter-Gather across non-contiguous local buffers.
  // IBV_SEND_INLINE is intentionally NOT set so the HCA DMA's from GPU memory.
  bool RdmaWriteSgl(uint64_t raddr, uint32_t rkey) {
    // TODO: 
  }

  bool Poll() {
    // TODO: 
  }

  // Client: fill each pinned-host segment with consecutive floats so the whole
  // sequence [0, 1, ..., N-1] is split across kNumSge non-contiguous buffers.
  // SGE[i] contains floats [i*seg_nf .. (i+1)*seg_nf - 1].
  void FillClientBuffers() {
    size_t seg_sz = buf_sz_ / kNumSge;
    size_t seg_nf = seg_sz / sizeof(float);
    for (int i = 0; i < kNumSge; i++) {
      float* dst = static_cast<float*>(bufs_[i]);
      size_t base = static_cast<size_t>(i) * seg_nf;
      for (size_t j = 0; j < seg_nf; j++) dst[j] = static_cast<float>(base + j);
    }
  }

  // Server: read the contiguous pinned-host buffer and verify the expected
  // float sequence [0, 1, ..., N-1] written by the client's SGL.
  bool VerifyServerBuffer() {
    size_t nf = buf_sz_ / sizeof(float);
    const float* src = static_cast<const float*>(bufs_[0]);
    for (size_t i = 0; i < nf; i++) {
      if (src[i] != static_cast<float>(i)) return false;
    }
    return true;
  }

 private:
  // ── buffer allocation ─────────────────────────────────────────────────────

  // Server: one contiguous page-aligned CPU allocation + one MR.
  // std::aligned_alloc gives memory in the standard CPU page table that
  // ibv_reg_mr can pin via pin_user_pages() on any platform.
  void AllocServerBuffer() {
    // TODO: 
  }

  // Client: kNumSge independent page-aligned CPU allocations, each registered
  // as its own MR.  Each MR has a unique addr/len/lkey, forming a true SGL of
  // non-contiguous local buffers gathered by one RDMA WRITE WR.
  void AllocClientBuffers() {
    size_t seg_sz = buf_sz_ / kNumSge;
    for (int i = 0; i < kNumSge; i++) {
      // TODO: 
    }
  }

  // ── QP helpers ────────────────────────────────────────────────────────────

  int PickGidIndex() {
    const char* env = std::getenv("RDMA_GID_INDEX");
    if (env) return std::atoi(env);
    ibv_port_attr pa{};
    if (ibv_query_port(ctx_, kPortNum, &pa) != 0) return 0;
    int best = 0;
    for (int i = 0; i < pa.gid_tbl_len && i < 16; i++) {
      union ibv_gid g{};
      if (ibv_query_gid(ctx_, kPortNum, i, &g) != 0) continue;
      bool zero = true;
      for (int j = 0; j < 16; j++) { if (g.raw[j]) { zero = false; break; } }
      if (zero) continue;
      if (g.raw[10] == 0xff && g.raw[11] == 0xff) return i;  // prefer IPv4-mapped
      if (!best) best = i;
    }
    return best;
  }

  void InitQp() {
    // TODO: 
  }

  ibv_context* ctx_ = nullptr;
  ibv_pd*      pd_  = nullptr;
  ibv_cq*      cq_  = nullptr;
  ibv_qp*      qp_  = nullptr;

  std::vector<void*>   bufs_;  // pinned host allocations (hipHostMalloc)
  std::vector<ibv_mr*> mrs_;       // one MR per pinned buffer

  size_t buf_sz_;
  bool   is_server_;
  int    gid_idx_;
  union ibv_gid gid_{};
};

// ─────────────────────────────────────────────────────────────────────────────
// Entry points
// ─────────────────────────────────────────────────────────────────────────────

static int RunServer(ibv_device* dev, uint16_t port) {
  TcpChannel ch;
  ch.Listen(port);
  ch.Accept();

  bool all_correct = true;
  size_t align = static_cast<size_t>(kNumSge) * sizeof(float);

  for (int t = 0; t < kNumTestSizes; t++) {
    size_t data_size = (kTestSizesMB[t] * 1024 * 1024 / align) * align;
    if (data_size == 0) data_size = align;

    RdmaEndpoint ep(dev, data_size, /*is_server=*/true);
    QpMetadata my = ep.GetMetadata();

    char meta[kMetaBytes];
    QpMetadataCodec::Pack(my, meta);
    ch.SendAll(meta, kMetaBytes);

    ch.RecvAll(meta, kMetaBytes);
    QpMetadata remote{};
    QpMetadataCodec::Unpack(meta, &remote);
    ep.Connect(remote);

    // Wait for the client to signal that the RDMA WRITE SGL is complete.
    char sig[4];
    ch.RecvAll(sig, 4);

    bool correct = ep.VerifyServerBuffer();
    all_correct &= correct;

    char result = correct ? 'P' : 'F';
    ch.SendAll(&result, 1);
  }

  std::cout << "{\"Correctness\":\"" << (all_correct ? "PASS" : "FAIL") << "\"}";
  return all_correct ? 0 : 1;
}

static int RunClient(ibv_device* dev,
                     const std::string& host, uint16_t port) {
  TcpChannel ch;
  ch.Connect(host, port);

  bool all_correct = true;
  size_t align = static_cast<size_t>(kNumSge) * sizeof(float);

  struct Metric { size_t data_mb; double gbps; double us; bool correct; };
  std::vector<Metric> metrics;

  for (int t = 0; t < kNumTestSizes; t++) {
    size_t data_size = (kTestSizesMB[t] * 1024 * 1024 / align) * align;
    if (data_size == 0) data_size = align;

    char meta[kMetaBytes];
    ch.RecvAll(meta, kMetaBytes);
    QpMetadata srv{};
    QpMetadataCodec::Unpack(meta, &srv);

    RdmaEndpoint ep(dev, data_size, /*is_server=*/false);
    QpMetadata my = ep.GetMetadata();
    QpMetadataCodec::Pack(my, meta);
    ch.SendAll(meta, kMetaBytes);
    ep.Connect(srv);

    // Fill kNumSge non-contiguous GPU buffers with the expected pattern.
    ep.FillClientBuffers();

    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK(ep.RdmaWriteSgl(srv.addr, srv.rkey), "RdmaWriteSgl");
    CHECK(ep.Poll(), "poll");
    auto t1 = std::chrono::high_resolution_clock::now();

    // Signal server that RDMA WRITE SGL is complete.
    ch.SendAll("DONE", 4);

    // Collect verification result from server.
    char result;
    ch.RecvAll(&result, 1);
    bool correct = (result == 'P');
    all_correct &= correct;

    double us   = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double gbps = (data_size * 8.0) / us / 1e3;
    metrics.push_back({kTestSizesMB[t], gbps, us, correct});
  }

  std::cout << "{\n"
            << "  \"Correctness\": \"" << (all_correct ? "PASS" : "FAIL") << "\",\n"
            << "  \"num_sge\": " << kNumSge << ",\n"
            << "  \"data_size_unit\": \"MB\",\n"
            << "  \"throughput_unit\": \"Gbps\",\n"
            << "  \"latency_unit\": \"us\",\n"
            << "  \"metrics\": [\n";
  for (size_t i = 0; i < metrics.size(); i++) {
    std::cout << "    {\n"
              << "      \"data_size\": " << metrics[i].data_mb << ",\n"
              << "      \"throughput_avg\": " << metrics[i].gbps << ",\n"
              << "      \"latency_avg\": " << metrics[i].us << "\n"
              << "    }" << (i + 1 < metrics.size() ? "," : "") << "\n";
  }
  std::cout << "  ]\n}";

  return all_correct ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " server|client --nic <idx> [--server <ip>] [--port <p>]\n";
    return 1;
  }

  bool is_server = std::string(argv[1]) == "server";
  bool is_client = std::string(argv[1]) == "client";

  int         nic  = 0;
  uint16_t    port = 9999;
  std::string host = "127.0.0.1";

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if      (a == "--nic"    && i + 1 < argc) nic  = std::atoi(argv[++i]);
    else if (a == "--port"   && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (a == "--server" && i + 1 < argc) host = argv[++i];
  }

  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs || nic >= ndev) return 1;

  try {
    if (is_server) return RunServer(devs[nic], port);
    if (is_client) return RunClient(devs[nic], host, port);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    std::cout << "{\"Correctness\":\"FAIL\"}";
    return 1;
  }

  return 0;
}
