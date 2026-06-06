/*
 * ref_rdma_atomic_fetch_add.cpp
 *
  * This is a reference implementation of atomic fetch-and-add using ibv_post_send
 *
 * Topology:
 *   - Server:
 *       * One shared uint64_t value in a single MR
 *       * Two RC QPs (ep0, ep1) both exposing the SAME addr/rkey
 *   - Client:
 *       * Two RC QPs (ep0, ep1), each connected to one server QP
 *       * Both issue fetch-and-add with add=1 against the SAME remote address
 *
 * Correctness rule:
  *   Two fetch-and-add with add=1 should return 0 and 1 in any order.
 *
 * Run:
 *   server: ./ref_rdma_atomic_fetch_add server --nic 0 [--port 9999]
 *   client: ./ref_rdma_atomic_fetch_add client --nic 0 --server <ip> [--port 9999]
 *
 * Output:
 *   Each process prints EXACTLY ONE JSON object.
 */

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void Die(const std::string &s)
{
  throw std::runtime_error(s);
}

#define CHECK(x, msg) \
  do                  \
  {                   \
    if (!(x))         \
      Die(msg);       \
  } while (0)

// qpn(4) + gid(16) + addr(8) + rkey(4) + gid_index(4)
static constexpr int kMetaBytes = 36;
static constexpr int kPortNum = 1;

struct QpMeta
{
  uint32_t qpn;
  union ibv_gid gid;
  uint64_t addr;
  uint32_t rkey;
  int gid_index;
};

static void PackMeta(const QpMeta &m, char *b)
{
  uint32_t u32;
  uint64_t u64;
  u32 = htonl(m.qpn);
  memcpy(b, &u32, 4);
  memcpy(b + 4, m.gid.raw, 16);
  u64 = htobe64(m.addr);
  memcpy(b + 20, &u64, 8);
  u32 = htonl(m.rkey);
  memcpy(b + 28, &u32, 4);
  u32 = htonl(m.gid_index);
  memcpy(b + 32, &u32, 4);
}

static void UnpackMeta(const char *b, QpMeta &m)
{
  uint32_t u32;
  uint64_t u64;
  memcpy(&u32, b, 4);
  m.qpn = ntohl(u32);
  memcpy(m.gid.raw, b + 4, 16);
  memcpy(&u64, b + 20, 8);
  m.addr = be64toh(u64);
  memcpy(&u32, b + 28, 4);
  m.rkey = ntohl(u32);
  memcpy(&u32, b + 32, 4);
  m.gid_index = ntohl(u32);
}

// ─────────────────────────────────────────────────────────────────────────────
// TCP helper
// ─────────────────────────────────────────────────────────────────────────────

class Tcp
{
public:
  ~Tcp()
  {
    if (fd_ >= 0)
      close(fd_);
    if (lfd_ >= 0)
      close(lfd_);
  }

  void Listen(uint16_t port)
  {
    lfd_ = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(lfd_ >= 0, "socket");
    int on = 1;
    setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons(port);
    CHECK(bind(lfd_, (sockaddr *)&a, sizeof(a)) == 0, "bind");
    CHECK(listen(lfd_, 1) == 0, "listen");
  }

  void Accept()
  {
    fd_ = accept(lfd_, nullptr, nullptr);
    CHECK(fd_ >= 0, "accept");
    close(lfd_);
    lfd_ = -1;
  }

  void Connect(const std::string &ip, uint16_t port)
  {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd_ >= 0, "socket");
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    CHECK(inet_pton(AF_INET, ip.c_str(), &a.sin_addr) == 1, "inet_pton");
    CHECK(connect(fd_, (sockaddr *)&a, sizeof(a)) == 0, "connect");
  }

  void Send(const void *p, size_t n)
  {
    const char *c = (const char *)p;
    while (n)
    {
      ssize_t s = send(fd_, c, n, 0);
      CHECK(s > 0, "send");
      c += s;
      n -= s;
    }
  }

  void Recv(void *p, size_t n)
  {
    char *c = (char *)p;
    while (n)
    {
      ssize_t s = recv(fd_, c, n, 0);
      CHECK(s > 0, "recv");
      c += s;
      n -= s;
    }
  }

private:
  int fd_ = -1;
  int lfd_ = -1;
};

// ─────────────────────────────────────────────────────────────────────────────
// RDMA endpoint
// ─────────────────────────────────────────────────────────────────────────────

class Endpoint
{
public:
  Endpoint(ibv_context *ctx, ibv_pd *pd, ibv_cq *cq)
      : ctx_(ctx), pd_(pd), cq_(cq)
  {
    //TODO: Create QP, allocate result buffer, register MR for result buffer
  }

  ~Endpoint()
  {
     //TODO:
  }

  // ─────────────────────────────────────────────────────────────
  // QP connect (RC + atomic)
  // ─────────────────────────────────────────────────────────────
  void Connect(const QpMeta &r)
  {
    //TODO:
  }

  // ─────────────────────────────────────────────────────────────
  // Atomic fetch-and-add using ibv_post_send (legacy path)
  // ─────────────────────────────────────────────────────────────
  bool FetchAdd(uint64_t remote_addr,
               uint32_t rkey,
               uint64_t add)
  {
     //TODO:
  }

  // ─────────────────────────────────────────────────────────────
  // CQ poll
  // ─────────────────────────────────────────────────────────────
  bool Poll()
  {
     //TODO:
  }

  uint64_t Result() const { return *result_; }

  QpMeta Meta(uint64_t addr, uint32_t rkey) const
  {
     //TODO:
  }

  // ─────────────────────────────────────────────────────────────
  // PickGidIndex 
  // ─────────────────────────────────────────────────────────────
  int PickGidIndex()
  {
     //TODO:
  }

private:
  ibv_context *ctx_;
  ibv_pd *pd_;
  ibv_cq *cq_;

  ibv_qp *qp_ = nullptr;

  union ibv_gid gid_{};
  int gid_index_ = 0;

  uint64_t *result_ = nullptr;
  ibv_mr *result_mr_ = nullptr;
};
// ─────────────────────────────────────────────────────────────────────────────
// Server
// ─────────────────────────────────────────────────────────────────────────────

static int RunServer(ibv_device *dev, uint16_t port)
{
  Tcp tcp;
  tcp.Listen(port);
  tcp.Accept();

  ibv_context *ctx = ibv_open_device(dev);
  CHECK(ctx, "open_device");
  ibv_pd *pd = ibv_alloc_pd(ctx);
  CHECK(pd, "alloc_pd");
  ibv_cq *cq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);
  CHECK(cq, "create_cq");

  uint64_t *shared = nullptr;
  CHECK(posix_memalign((void**)&shared, 8, sizeof(uint64_t)) == 0,
        "posix_memalign");
  *shared = 0;

  ibv_mr *mr = ibv_reg_mr(pd, shared, sizeof(uint64_t),
                          IBV_ACCESS_LOCAL_WRITE |
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_ATOMIC);
  CHECK(mr, "reg_mr shared");

  {
    Endpoint ep0(ctx, pd, cq);
    Endpoint ep1(ctx, pd, cq);

    char buf[kMetaBytes];
    PackMeta(ep0.Meta((uint64_t)shared, mr->rkey), buf);
    tcp.Send(buf, kMetaBytes);
    PackMeta(ep1.Meta((uint64_t)shared, mr->rkey), buf);
    tcp.Send(buf, kMetaBytes);

    QpMeta c0{}, c1{};
    tcp.Recv(buf, kMetaBytes);
    UnpackMeta(buf, c0);
    tcp.Recv(buf, kMetaBytes);
    UnpackMeta(buf, c1);

    ep0.Connect(c0);
    ep1.Connect(c1);
    tcp.Recv(buf, 2); // wait "OK"
  }
  std::cout << "{\"Correctness\":\"PASS\"}" << std::endl;

  ibv_dereg_mr(mr);
  free(shared);
  ibv_destroy_cq(cq);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Client
// ─────────────────────────────────────────────────────────────────────────────

static int RunClient(ibv_device *dev, const std::string &ip, uint16_t port)
{
  Tcp tcp;
  tcp.Connect(ip, port);

  char buf[kMetaBytes];
  QpMeta s0{}, s1{};
  tcp.Recv(buf, kMetaBytes);
  UnpackMeta(buf, s0);
  tcp.Recv(buf, kMetaBytes);
  UnpackMeta(buf, s1);

  ibv_context *ctx = ibv_open_device(dev);
  CHECK(ctx, "open_device");
  ibv_pd *pd = ibv_alloc_pd(ctx);
  CHECK(pd, "alloc_pd");
  ibv_cq *cq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);
  CHECK(cq, "create_cq");
  struct BenchPoint
  {
    double data_size_mb;
    double throughput_items_per_sec;
    double latency_us;
  };
  std::vector<BenchPoint> bench;
  bool ok;

  {
    Endpoint ep0(ctx, pd, cq);
    Endpoint ep1(ctx, pd, cq);

    PackMeta(ep0.Meta(0, 0), buf);
    tcp.Send(buf, kMetaBytes);
    PackMeta(ep1.Meta(0, 0), buf);
    tcp.Send(buf, kMetaBytes);
    ep0.Connect(s0);
    ep1.Connect(s1);

    // Issue one fetch-and-add to test correctness
    ep0.FetchAdd(s0.addr, s0.rkey, 0);
    ep0.FetchAdd(s1.addr, s1.rkey, 1);
    ep1.FetchAdd(s1.addr, s1.rkey, 1);
    ep0.Poll();
    ep0.Poll();
    ep1.Poll();

    uint64_t r0 = ep0.Result();
    uint64_t r1 = ep1.Result();
    ok =   ((r0 == 0 && r1 == 1) || (r0 == 1 && r1 == 0));
    ep0.FetchAdd(s0.addr, s0.rkey, 1);
    ep0.Poll();

    // ── Warmup ────────────────────────────────────────────────────────────
    for (int i = 0; i < 100; i++)
    {
      ep0.FetchAdd(s0.addr, s0.rkey, 1);
      ep0.Poll();
    }

    // ── Benchmark ─────────────────────────────────────────────────────────
    // Sweep two total-data sizes (1 MB and 2 MB worth of 8-byte atomics).
    // Because max_rd_atomic = 1, operations are sequential, so latency and
    // throughput are derived from the same elapsed time.
    //   latency_us    = elapsed_s / iters * 1e6
    //   throughput_Gbps = iters / elapsed_s / 1e9
    for (int iters : std::vector<int>{131072})
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; i++)
      {
        ep0.FetchAdd(s0.addr, s0.rkey, 1);
        ep0.Poll();
        uint64_t r0 = ep0.Result();
      }
      auto t1 = std::chrono::high_resolution_clock::now();

      double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
      double latency_us    = elapsed_s / iters * 1e6;
      double throughput_items_per_sec =
          static_cast<double>(iters) / elapsed_s;
      bench.push_back({sizeof(uint64_t), throughput_items_per_sec, latency_us});
    }
  }

  tcp.Send("OK", 2);

  // ── JSON output ───────────────────────────────────────────────────────────
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "{\n"
            << "  \"Correctness\": \"" << (ok ? "PASS" : "FAIL") << "\",\n"
            << "  \"data_size_unit\": \"B\",\n"
            << "  \"throughput_unit\": \"items_per_sec\",\n"  
            << "  \"latency_unit\": \"us\",\n"
            << "  \"metrics\": [\n";
  for (size_t i = 0; i < bench.size(); i++)
  {
    std::cout << "    {\"data_size\": " << bench[i].data_size_mb
              << ", \"throughput_avg\": " << bench[i].throughput_items_per_sec
              << ", \"latency_avg\": " << bench[i].latency_us << "}";
    if (i + 1 < bench.size())
      std::cout << ",";
    std::cout << "\n";
  }
  std::cout << "  ]\n"
            << "}" << std::endl;

  ibv_destroy_cq(cq);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  return ok ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
  if (argc < 2)
    return 1;

  bool is_server = std::string(argv[1]) == "server";
  bool is_client = std::string(argv[1]) == "client";

  int nic = 0;
  uint16_t port = 9999;
  std::string ip = "127.0.0.1";

  for (int i = 2; i < argc; i++)
  {
    std::string a = argv[i];
    if (a == "--nic" && i + 1 < argc)
      nic = atoi(argv[++i]);
    else if (a == "--port" && i + 1 < argc)
      port = atoi(argv[++i]);
    else if (a == "--server" && i + 1 < argc)
      ip = argv[++i];
  }

  int n = 0;
  ibv_device **devs = ibv_get_device_list(&n);
  if (!devs || nic >= n)
    return 1;

  int rc = 0;
  // try
  // {
  if (is_server)
    rc = RunServer(devs[nic], port);
  else if (is_client)
    rc = RunClient(devs[nic], ip, port);
  // }
  // catch (const std::exception &e)
  // {
  //   std::cerr << e.what() << std::endl;
  //   std::cout << "{\"Correctness\":\"FAIL\"}" << std::endl;
  //   rc = 1;
  // }

  ibv_free_device_list(devs);
  return rc;
}