/*
empty_rdma_ibv_wr_bind.cpp

Purpose:
  Empty template for RDMA Memory Window (MW) bind and revocation using
  ibv_wr_bind_mw and ibv_wr_local_inv on GPU-registered memory.
  Core implementation has been removed and replaced with // TODO.
  Fill in each // TODO to reproduce the reference implementation.

How to build:
  g++ -std=c++17 -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
      empty_rdma_ibv_wr_bind.cpp -o empty_rdma_ibv_wr_bind -libverbs -lcudart

How to run:
  Terminal 1 (server): SERVER_GPU=0 ./empty_rdma_ibv_wr_bind
  Terminal 2 (client): CLIENT_GPU=0 ./empty_rdma_ibv_wr_bind <server_ip>
*/

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// config

constexpr int    NUM_NICS   = 4;
constexpr int    NUM_GPUS   = 8;
constexpr int    IB_PORT    = 1;
constexpr int    GID_INDEX  = 0;    // 0 for IB; 3 for RoCEv2
constexpr int    TCP_PORT   = 12345;
constexpr int    CHUNK_NUM  = 4;
constexpr size_t CHUNK_SIZE = 1u << 20;   // 1 MB
constexpr size_t TOTAL_SIZE = CHUNK_NUM * CHUNK_SIZE;

static bool kDebug = false;
#define DPRINTF(...) do { if (kDebug) printf(__VA_ARGS__); } while (0)

struct ConnInfo  { uint32_t qpn; uint16_t lid; uint8_t gid[16]; };
struct ChunkInfo { uint64_t addr; uint32_t rkey; };
struct TestResult { bool pass; double latency_us; };

static void printJson(bool pass, double latency_us = -1.0) {
    if (latency_us >= 0.0) {
        printf("{\n"
               "  \"Correctness\": \"%s\",\n"
               "  \"latency_unit\": \"us\",\n"
               "  \"metrics\": [\n"
               "    {\"latency_avg\": %.2f}\n"
               "  ]\n"
               "}\n",
               pass ? "PASS" : "FAIL", latency_us);
    } else {
        printf("{\n  \"Correctness\": \"%s\"\n}\n",
               pass ? "PASS" : "FAIL");
    }
}

#define CUDA_CHECK(cmd)                                                   \
  do {                                                                    \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                      \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// ---------------------------------------------------------------------------
// RDMANode: shared RDMA infrastructure for Server and Client
// ---------------------------------------------------------------------------
class RDMANode {
protected:
    ibv_context* ctx_ = nullptr;
    ibv_pd*      pd_  = nullptr;
    ibv_cq*      cq_  = nullptr;
    ibv_qp*      qp_  = nullptr;
    ibv_mr*      mr_  = nullptr;
    void*        buf_ = nullptr;
    int          fd_  = -1;
    int          gpu_id_ = 0;

    // Open the first mlx5 NIC found and store its context in ctx_
    void openNic() {
        // TODO
    }

    // Allocate PD and CQ, then create an RC QP with MW-bind capabilities
    void createResources() {
        // TODO
    }

    // Exchange QP info over TCP and move QP through INIT -> RTR -> RTS
    void connectQp(bool is_server) {
        // TODO
    }

    // Spin-poll the CQ until one completion arrives
    ibv_wc pollOne() {
        // TODO
        ibv_wc wc = {};
        return wc;
    }

    // Release RDMA and CUDA resources common to both roles
    void teardown() {
        // TODO
    }
};

// ---------------------------------------------------------------------------
// Server: receiver — owns GPU buffer, grants/revokes MW access to client
// ---------------------------------------------------------------------------
class Server : public RDMANode {
    ibv_mw* mw_[CHUNK_NUM] = {};

public:
    // Allocate GPU buffer, open NIC, create RDMA resources, alloc MWs
    void setup(int gpu_id) {
        // TODO
    }

    // Listen for TCP connection, accept, then bring QP to RTS
    void connect() {
        // TODO
    }

    // Bind MW[i] with remote-write access and send ChunkInfo to client
    bool grantAccess(int i) {
        // TODO
        return false;
    }

    // Wait for client's "write done" signal
    bool waitWriteDone() {
        // TODO
        return false;
    }

    // Invalidate MW[i] via local-inv WR, then notify client
    bool revokeAccess(int i) {
        // TODO
        return false;
    }

    // Return pointer to the start of chunk i in the GPU buffer
    void* getChunkAddr(int i) {
        // TODO
        return nullptr;
    }

    // Release all server resources
    void teardown() {
        // TODO
    }
};

// ---------------------------------------------------------------------------
// Client: sender — writes into server's GPU buffer using granted rkeys
// ---------------------------------------------------------------------------
class Client : public RDMANode {
public:
    // Allocate GPU buffer, open NIC, create RDMA resources
    void setup(int gpu_id) {
        // TODO
    }

    // TCP connect to server, then bring QP to RTS
    void connect(const char* ip) {
        // TODO
    }

    // Receive ChunkInfo (addr + rkey) sent by server for chunk i
    ChunkInfo receiveChunkInfo(int i) {
        // TODO
        return {};
    }

    // Post an RDMA write for chunk i using the given addr/rkey; return wc status
    ibv_wc_status writeChunk(int i, uint64_t addr, uint32_t rkey) {
        // TODO
        return IBV_WC_GENERAL_ERR;
    }

    // Signal server that the RDMA write is done
    bool signalWriteDone() {
        // TODO
        return false;
    }

    // Wait for server's revocation notification
    bool waitRevocation() {
        // TODO
        return false;
    }

    // Release all client resources
    void teardown() {
        // TODO
    }
};

// ---------------------------------------------------------------------------
// runTest: initialize server or client based on argv, run the MW-bind test
// ---------------------------------------------------------------------------
static TestResult runTest(int argc, char* argv[]) {
    int gpu_id = 0;
    const char* env = getenv(argc == 1 ? "SERVER_GPU" : "CLIENT_GPU");
    if (env) gpu_id = atoi(env);

    if (argc == 1) {
        Server server;
        server.setup(gpu_id);
        server.connect();

        bool ok = true;
        auto t0 = std::chrono::high_resolution_clock::now();
        ok = ok && server.grantAccess(0);
        ok = ok && server.waitWriteDone();

        // Verify client wrote 0xAB into chunk 0
        uint8_t* host = new uint8_t[CHUNK_SIZE];
        CUDA_CHECK(cudaMemcpy(host, server.getChunkAddr(0), CHUNK_SIZE, cudaMemcpyDeviceToHost));
        bool written = true;
        for (size_t j = 0; j < CHUNK_SIZE; j++) {
            if (host[j] != 0xAB) { written = false; break; }
        }
        delete[] host;
        DPRINTF("[Server] chunk0  content check: %s\n", written ? "PASS" : "FAIL");
        ok = ok && written;

        ok = ok && server.revokeAccess(0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        DPRINTF("\n[Server] done\n");
        server.teardown();
        return {ok, latency_us};
    } else {
        const char* ip = argv[1];
        if (argc >= 3) gpu_id = atoi(argv[2]);

        Client client;
        client.setup(gpu_id);
        client.connect(ip);

        bool ok = true;
        bool blocked_once = false;
        auto t0 = std::chrono::high_resolution_clock::now();

        ChunkInfo ci = client.receiveChunkInfo(0);

        // Write with valid rkey — expect SUCCESS
        ibv_wc_status s = client.writeChunk(0, ci.addr, ci.rkey);
        if (s != IBV_WC_SUCCESS) ok = false;
        DPRINTF("[Client] chunk0  write (valid rkey):   %s\n", ibv_wc_status_str(s));

        ok = ok && client.signalWriteDone();
        ok = ok && client.waitRevocation();

        // Write with revoked rkey — expect REM_ACCESS_ERR or RETRY_EXC_ERR
        s = client.writeChunk(0, ci.addr, ci.rkey);

        auto t1 = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        bool blocked = (s == IBV_WC_REM_ACCESS_ERR || s == IBV_WC_RETRY_EXC_ERR);
        blocked_once = blocked_once || blocked;
        DPRINTF("[Client] chunk0  write (revoked rkey): %s  %s\n\n",
               ibv_wc_status_str(s), blocked ? "BLOCKED" : "UNEXPECTED");

        DPRINTF("[Client] done\n");
        client.teardown();
        return {ok && blocked_once, latency_us};
    }
}

int main(int argc, char* argv[]) {
    TestResult result = runTest(argc, argv);
    printJson(result.pass, result.latency_us);
    return result.pass ? 0 : 1;
}
