/* Description:
Implement a class which provides recv/send functions using iB UD recv/send. Communication should happen on two different nic on one node.
Exchange the data on both CPU and GPU memory.

Connectionless: no pairing. Each send names the destination via AH + remote_qpn + qkey.
Unreliable: the network does not guarantee delivery.
No ordering guarantee: datagrams can be delivered out-of-order.
No retransmissions: if it drops, it drops.
Bounded size: one WR corresponds to one datagram bounded by MTU (no automatic segmentation like RC).
Receiver must have posted buffers: if there’s no recv WQE available, the packet is dropped.
*/


#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define DEFAULT_ITERATIONS 100
#define gpuSuccess        hipSuccess
#define gpuError_t        hipError_t
#define gpuGetErrorString hipGetErrorString
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
static constexpr int kNumSge     = 1; 
static constexpr int kMaxSendSge = kNumSge;  // QP must support kNumSge SGEs per WR
static constexpr int kMaxRecvSge = 1;
static constexpr int kMaxCqe     = 512;
static constexpr uint32_t kQKey = 0x11111111; // arbitrary non-zero qkey for UD
static constexpr size_t kGrhBytes = 40; // GRH is 40 bytes, so payload must be at least this big to test GRH handling
static int g_gid_index = -1;

static constexpr size_t kUdMtuBytes = 4096;

static constexpr int kMaxSendWr = 512;
static constexpr int kMaxRecvWr = 512;

static const size_t kTestSizesB[] = {
    64, 256, 512, 1024, 1024 * 2, 1024 * 4,           // sub-MTU and exactly-MTU
    1024 * 8, 1024 * 64, 1024 * 128, 1024 * 256,  
    // 1024 * 512, 1 * 1024 * 1024,     these two wont work 
};
static constexpr int kNumTestSizes = sizeof(kTestSizesB) / sizeof(kTestSizesB[0]);

// Wire format: qpn(4) + gid(16) + gid_index(4) + qkey(4) = 28 bytes
static constexpr int kMetaBytes = 28;

// Known fill pattern: every byte is 0xAB so we can verify on the server side.
static constexpr uint8_t kFillPattern = 0xAB;

// Global mem type flag — set from --mem-type cpu|gpu argument.
static bool g_use_gpu = true;

// ─────────────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────────────
// How many UD datagrams are needed to carry `total` bytes?
static inline size_t NumChunks(size_t total) {
  return (total + kUdMtuBytes - 1) / kUdMtuBytes;
}

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

#define CHECK_HIP(st, where) \
  do { \
    if ((st) != hipSuccess) { \
      std::cerr << (where) << " failed: " << hipGetErrorString(st) << "\n"; \
      std::exit(EXIT_FAILURE); \
    } \
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
      if (n == 0) 
        throw std::runtime_error("recv: connection closed by peer");
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
  int           gid_index;
  uint32_t      qkey;
};
// qpn(4) + gid(16) + gid_index(4) + qkey(4) = 28 bytes
class QpMetadataCodec {
 public:
  static void Pack(const QpMetadata& m, char* buf) {
    uint32_t u32 = htonl(m.qpn);
    std::memcpy(buf,      &u32,      4);
    std::memcpy(buf + 4,  m.gid.raw, 16);
    int32_t i32 = htonl(m.gid_index);
    std::memcpy(buf + 20, &i32,      4);
    u32 = htonl(m.qkey);
    std::memcpy(buf + 24, &u32,      4);
  }

  static void Unpack(const char* buf, QpMetadata* m) {
    uint32_t u32; int32_t i32;
    std::memcpy(&u32, buf,      4); m->qpn = ntohl(u32);
    std::memcpy(m->gid.raw, buf + 4, 16);
    std::memcpy(&i32, buf + 20, 4); m->gid_index = ntohl(i32);
    std::memcpy(&u32, buf + 24, 4); m->qkey = ntohl(u32);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// RDMA endpoint
// ─────────────────────────────────────────────────────────────────────────────

class RdmaEndpoint_UD {
 public:
  RdmaEndpoint_UD(ibv_device* dev, size_t buf_sz, bool is_server)
      : buf_sz_(buf_sz), is_server_(is_server), use_gpu_(g_use_gpu) {
        // TODO
  }

  ~RdmaEndpoint_UD() {
    // TODO
  }

  QpMetadata GetMetadata() const {
    // TODO
  }

void Connect(const QpMetadata& remote) {
  // TODO
}

bool PostAllRecvs() {
  // TODO
}

bool Send() {
  // TODO
}

bool PollAllRecvs() {
  // TODO
}

// Fill the client's GPU send buffer with kFillPattern.
// hipMemset sets every byte to the value — no CPU involvement.
  void FillClientBuffers() {
    // TODO
  }

// ── buffer allocation ────────────────────────────────────────────────────
  // want to support both CPU and GPU memory backed recv and send over RDMA
  // Server recv buffer layout:
  //   [ GRH (kGrhBytes=40 B) | payload (buf_sz_ B) ]
  //   Total = kGrhBytes + buf_sz_
  //   The NIC writes the GRH into the first 40 bytes, then the payload follows.
  //   VerifyServerBuffer() reads from gpu_buf_ + kGrhBytes.
  //
  // Client send buffer layout:
  //   [ payload (buf_sz_ B) ]
  //   No GRH prefix needed on the send side; the NIC generates the GRH itself
  //   from the AH. The SGE points directly at gpu_buf_.
  //
  // hipMemset zeros the allocation before use; no hipDeviceSynchronize needed
  // after hipMalloc since malloc doesn't launch any work on the GPU stream.
  bool VerifyServerBuffer() {
    // TODO
  }

 private:

  void AllocServerBuffer() {
    // TODO
  }

  void AllocClientBuffers() {
    // TODO
  }

  bool PollOneSend() {
    // TODO
  }

  int PickGidIndex() {
    // TODO
  }

  void InitQp() {
    // TODO
  }

  ibv_context* ctx_ = nullptr;
  ibv_pd*      pd_  = nullptr;
  ibv_cq*      cq_  = nullptr;
  ibv_qp*      qp_  = nullptr;
  ibv_ah*      ah_  = nullptr;

  void*   buf_ = nullptr;
  ibv_mr* mr_  = nullptr;
  std::vector<uint8_t> payload_;

  size_t buf_sz_;
  bool   is_server_;
  bool   use_gpu_;      // true = hipMalloc, false = aligned_alloc
  int    gid_idx_;
  union ibv_gid gid_{};
  uint32_t remote_qpn_ = 0;
  uint32_t remote_qkey_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Entry points
// ─────────────────────────────────────────────────────────────────────────────

static int RunServer(ibv_device* dev, uint16_t port) {
  TcpChannel ch;
  ch.Listen(port);
  ch.Accept();

  bool all_correct = true;

  for (int t = 0; t < kNumTestSizes; t++) {
    size_t data_size = kTestSizesB[t];
    size_t n_chunks  = NumChunks(data_size);

    std::cerr << "\n[Server] === size=" << data_size << "B chunks=" << n_chunks << " ===\n";

    RdmaEndpoint_UD ep(dev, data_size, /*is_server=*/true);

    char meta[kMetaBytes];
    QpMetadataCodec::Pack(ep.GetMetadata(), meta);
    ch.SendAll(meta, kMetaBytes);

    ch.RecvAll(meta, kMetaBytes);
    QpMetadata remote{};
    QpMetadataCodec::Unpack(meta, &remote);
    ep.Connect(remote);

    CHECK(ep.PostAllRecvs(), "PostAllRecvs");

    char ready = 'R';
    ch.SendAll(&ready, 1);

    CHECK(ep.PollAllRecvs(), "PollAllRecvs");

    bool correct = ep.VerifyServerBuffer();
    all_correct &= correct;
    std::cerr << "[Server] verify=" << (correct ? "PASS" : "FAIL") << "\n";

    char result = correct ? 'P' : 'F';
    ch.SendAll(&result, 1);
  }

  std::cout << "{\"Correctness\":\"" << (all_correct ? "PASS" : "FAIL") << "\"}\n";
  return all_correct ? 0 : 1;
}

static int RunClient(ibv_device* dev, const std::string& host, uint16_t port) {
  TcpChannel ch;
  ch.Connect(host, port);

  bool all_correct = true;

  struct Metric { size_t data_b; double mbps; double us; bool correct; };
  std::vector<Metric> metrics;

  for (int t = 0; t < kNumTestSizes; t++) {
    size_t data_size = kTestSizesB[t];
    size_t n_chunks  = NumChunks(data_size);

    std::cerr << "\n[Client] === size=" << data_size << "B chunks=" << n_chunks << " ===\n";

    char meta[kMetaBytes];
    ch.RecvAll(meta, kMetaBytes);
    QpMetadata srv{};
    QpMetadataCodec::Unpack(meta, &srv);

    RdmaEndpoint_UD ep(dev, data_size, /*is_server=*/false);
    QpMetadataCodec::Pack(ep.GetMetadata(), meta);
    ch.SendAll(meta, kMetaBytes);
    ep.Connect(srv);

    ep.FillClientBuffers();

    char ready;
    ch.RecvAll(&ready, 1);
    CHECK(ready == 'R', "expected ready signal from server");

    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK(ep.Send(), "Send failed");
    auto t1 = std::chrono::high_resolution_clock::now();

    char result;
    ch.RecvAll(&result, 1);
    bool correct = (result == 'P');
    all_correct &= correct;

    double us   = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double mbps = (data_size * 8.0) / us;
    metrics.push_back({data_size, mbps, us, correct});

    std::cerr << "[Client] us=" << us << " mbps=" << mbps
              << " correct=" << (correct ? "yes" : "NO") << "\n";
  }

  std::cout << "{\n"
            << "  \"Correctness\": \"" << (all_correct ? "PASS" : "FAIL") << "\",\n"
            << "  \"mem_type\": \"" << (g_use_gpu ? "gpu" : "cpu") << "\",\n"
            << "  \"ud_mtu_bytes\": " << kUdMtuBytes << ",\n"
            << "  \"data_size_unit\": \"B\",\n"
            << "  \"throughput_unit\": \"Mbps\",\n"
            << "  \"latency_unit\": \"us\",\n"
            << "  \"metrics\": [\n";
  for (size_t i = 0; i < metrics.size(); i++) {
    std::cout << "    {\n"
              << "      \"data_size\": "      << metrics[i].data_b << ",\n"
              << "      \"throughput_avg\": " << metrics[i].mbps   << ",\n"
              << "      \"latency_avg\": "    << metrics[i].us     << ",\n"
              << "      \"correct\": "        << (metrics[i].correct ? "true" : "false") << "\n"
              << "    }" << (i + 1 < metrics.size() ? "," : "") << "\n";
  }
  std::cout << "  ]\n}\n";

  return all_correct ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " server|client --nic <idx> [--server <ip>] [--port <p>]"
              << " [--gid-index <n>] [--mem-type cpu|gpu]\n";
    return 1;
  }

  bool is_server = (std::string(argv[1]) == "server");
  bool is_client = (std::string(argv[1]) == "client");
  if (!is_server && !is_client) {
    std::cerr << "first arg must be server or client\n";
    return 1;
  }

  int         nic  = 0;
  uint16_t    port = 9999;
  std::string host = "127.0.0.1";

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--nic" && i + 1 < argc) {
      nic = std::atoi(argv[++i]);
    } else if (a == "--port" && i + 1 < argc) {
      port = static_cast<uint16_t>(std::atoi(argv[++i]));
    } else if (a == "--server" && i + 1 < argc) {
      host = argv[++i];
    } else if (a == "--gid-index" && i + 1 < argc) {
      g_gid_index = std::atoi(argv[++i]);
    } else if (a == "--mem-type" && i + 1 < argc) {
      std::string mt = argv[++i];
      if (mt == "gpu")       g_use_gpu = true;
      else if (mt == "cpu")  g_use_gpu = false;
      else { std::cerr << "--mem-type must be cpu or gpu\n"; return 1; }
    } else {
      std::cerr << "unknown/invalid arg: " << a << "\n";
      return 1;
    }
  }

  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs || ndev == 0) {
    std::cerr << "no RDMA devices found\n";
    return 1;
  }
  if (nic < 0 || nic >= ndev) {
    std::cerr << "bad --nic index " << nic << " (ndev=" << ndev << ")\n";
    return 1;
  }

  std::cerr << (is_server ? "server" : "client")
            << ": device=" << ibv_get_device_name(devs[nic])
            << " port=" << port
            << " mem_type=" << (g_use_gpu ? "gpu" : "cpu")
            << (is_client ? (std::string(" server=") + host) : "")
            << "\n";

  int rc = 1;
  try {
    if (is_server) rc = RunServer(devs[nic], port);
    else           rc = RunClient(devs[nic], host, port);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    rc = 1;
  }

  ibv_free_device_list(devs);
  return rc;
}