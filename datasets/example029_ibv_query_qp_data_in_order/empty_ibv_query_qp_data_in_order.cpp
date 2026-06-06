/*
Use ibv_query_qp_data_in_order to investigate the data ordering guarantees provided by different QP transport types and RDMA operations.

The goal of this task is to build a small program that queries whether data arrival is guaranteed to be in order for different combinations of:

QP transport types (e.g., RC, UC, UD)
RDMA operation opcodes (e.g., IBV_WR_SEND, IBV_WR_RDMA_WRITE, IBV_WR_RDMA_WRITE_WITH_IMM)
You should:

Create several QPs with different transport types (RC, UC, and UD if supported by the device).
For each QP, call ibv_query_qp_data_in_order() with different opcodes.
Check the returned value (1 = ordered, 0 = unordered).
This experiment helps understand how different RDMA transports enforce (or do not enforce) write ordering, 
which is important when designing communication protocols on top of RDMA.
 */
 
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#if defined(__linux__)
#include <endian.h>
#else
#define be64toh(x) __builtin_bswap64(x)
#define htobe64(x) __builtin_bswap64(x)
#endif

// ── constants ────────────────────────────────────────────────────────────
static constexpr bool kLLM_IMPLEMENTATION_COMPLETE = false; 
// finish Flag. change this to true if you are done with ALL the implementation.
static constexpr int PORT_NUM       = 1;
static constexpr int MAX_SEND_WR    = 256;
static constexpr int MAX_RECV_WR    = 256;
static constexpr int MAX_SEND_SGE   = 1;
static constexpr int MAX_RECV_SGE   = 1;
static constexpr int MAX_CQE        = 512;
static constexpr int MIN_RNR_TIMER  = 12;
static constexpr int TIMEOUT        = 14;
static constexpr int RETRY_CNT      = 7;
static constexpr int MAX_RD_ATOMIC  = 1;

static int g_gid_index = -1;

#define CHECK_ERRNO(cond, msg) \
    do { if (!(cond)) { perror(msg); throw std::runtime_error(std::string(msg) + ": " + strerror(errno)); } } while(0)
#define CHECK(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(msg); } while(0)


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
    RdmaEndpoint(ibv_device* dev, ibv_qp_type qp_type){
        // TODO
    }

    ~RdmaEndpoint() {
        // TODO
    }

    int queryDataInOrder(ibv_wr_opcode opcode) {
        // TODO
    }

private:
    int pickGidIndex() {
        // TODO
    }

    void initQP(ibv_qp_type qp_type) {
        // TODO
    }

    ibv_context* ctx_ = nullptr;
    ibv_pd*      pd_  = nullptr;
    ibv_cq*      cq_  = nullptr;
    ibv_qp*      qp_  = nullptr;
    int          gid_idx_;
    union ibv_gid gid_;
};

// ── test runner ─────────────────────────────────────────────────────────

struct QueryResult {
    const char* transport;
    const char* opcode;
    int in_order;
};

struct TestResult {
    bool correct;
    double latency_avg_us;
};

static std::vector<QueryResult> runTest(ibv_device* dev) {
    struct TransportEntry { ibv_qp_type type; const char* name; };
    struct OpcodeEntry    { ibv_wr_opcode opcode; const char* name; };

    TransportEntry transports[] = {
        {IBV_QPT_RC, "RC"},
        {IBV_QPT_UC, "UC"},
        {IBV_QPT_UD, "UD"},
    };
    OpcodeEntry opcodes[] = {
        {IBV_WR_SEND,                "SEND"},
        {IBV_WR_RDMA_WRITE,          "RDMA_WRITE"},
        {IBV_WR_RDMA_WRITE_WITH_IMM, "RDMA_WRITE_WITH_IMM"},
        // more can be implemented if needed
    };

    std::vector<QueryResult> results;
    for (auto& t : transports) {
        RdmaEndpoint ep(dev, t.type);
        for (auto& o : opcodes) {
            int r = ep.queryDataInOrder(o.opcode);
            results.push_back({t.name, o.name, r});
            std::cerr << t.name << " + " << o.name << " => " << (r ? "IN_ORDER" : "UNORDERED") << "\n";
        }
    }
    return results;
}

static void printResultJson(const std::vector<QueryResult>& results) {
    std::cout << "{\n";
    std::cout << "  \"Correctness\": \"PASS\",\n";
    std::cout << "  \"metrics\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        std::cout << "    {\"transport\": \"" << r.transport
                  << "\", \"opcode\": \"" << r.opcode
                  << "\", \"in_order\": " << r.in_order << "}";
        if (i + 1 < results.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ]\n}\n";
}

// ── main ────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    bool server = false, client = false;
    int nic_idx = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--nic"        && i+1 < argc)      nic_idx = std::atoi(argv[++i]);
        else if (a == "--gid-index"  && i+1 < argc)      g_gid_index = std::atoi(argv[++i]);
    }

    auto devs = getDevices();
    if (devs.empty()) { std::cerr << "no RDMA devices\n"; return 1; }
    if (nic_idx >= (int)devs.size()) { std::cerr << "bad NIC index\n"; return 1; }

    std::cerr << ": device=" << ibv_get_device_name(devs[nic_idx]) << "\n";

    try {
        auto result = runTest(devs[nic_idx]);
        printResultJson(result);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}