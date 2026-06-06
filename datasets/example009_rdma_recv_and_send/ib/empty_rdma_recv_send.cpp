
/*
 * RDMA NIC PingPong API
 *
 * This program implement a class which provides recv/send functions using iB RC RC recv/send. 
 * Communication should happen on two different nic on one node.
 *
 * The RdmaNicInfo class provides methods to discover and query all RDMA
 * devices on the system, printing both human-readable summaries and
 * deterministic METRICS_JSON output for automated comparison.
 *
 * YOUR TASK: Implement the 5 methods marked with TODO below. 
 * All output methods, data structures, includes, and main() are provided.
 * Do NOT modify the output methods or data structures.
 */

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>


static constexpr bool kLLM_IMPLEMENTATION_COMPLETE = false; 
// finish Flag. change this to true if you are done with ALL the implementation.

class RcChannel {
public:
    RcChannel(const std::string& device_name, uint8_t port_num, int gid_index)
        : device_name_(device_name),
          port_num_(port_num),
          gid_index_(gid_index) {
        // TODO
        // this function initilizes the channel
        // first it opens RDMA device by name. 
        // Then it query port info and allocate pd. 
        // Then it creates CQ. 
        // After that, It allocates a page-alighned internal buffer zone, 
        // this zone needs to be registered as MR.
        // create RC Queue Pair bound to that CQ. 
        // Transition QP to INIT state.
        // For the peer to send, pre-post necessary info.
        // Throw any exception on failure. 
    }

    
    void connect(const std::string& peer_ip) {
        // TODO
        // Client-mode connect: connect to peer_ip:TCP_PORT and exchange dests.
        // If peer_ip is empty string, run server-mode: listen and accept one connection.
    }

    void send(const void* buf, size_t size) {
        // TODO
        // this function copies user data to internal registered buffer
        // Post send, then poll until send completion.
        // Throw any exception on failure. 
    }

    void recv(void* buf, size_t size) {
        // TODO
        // this function polls until recv completion,
        // then copy data from internal registered buffer to user buf.
        // After that, post one more receive to replenish the RX queue.
        // Throw any exception on failure.
    }

    ~RcChannel() {
        // TODO
        // Destroy QP, CQ, MR, PD, close device, free buffer, free device list as needed.
        // Don't throw exceptions in destructor; best-effort cleanup.
    }

    // for some private variables:
    // Use TCP port 18515 for handshake
    // Keep rx_depth >= 128 receives posted
    // poll 8 CQEs per call
    // Use MTU 1024 or use port_attr.active_mtu
    // Internal buffer 1MiB, align to page size

};


static void parse_optional_args(int argc, char** argv,
                                size_t& msg_size) {
    // Accept:
    //   --msg-size <bytes>
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--msg-size") == 0 && i + 1 < argc) {
            msg_size = static_cast<size_t>(std::stoull(argv[i + 1]));
            i++;
        }
    }
}

struct MetricRow {
    size_t msg_size;
    double bandwidth_gbps;
    double latency_usec;
};

static std::pair<bool, std::vector<MetricRow>> runTest(
    RcChannel& ch,
    const std::vector<size_t>& msg_sizes,
    size_t warmup_iters,
    size_t iters) {
    
    std::vector<MetricRow> rows;
    bool overall_pass = true;
    
    for (size_t msg_size : msg_sizes) {
        // Allocate test buffers
        std::vector<uint8_t> send_buf(msg_size);
        std::vector<uint8_t> recv_buf(msg_size);
        
        // Fill send buffer with known pattern for verification
        std::mt19937 rng(msg_size);
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (auto& b : send_buf) {
            b = dist(rng);
        }
        
        try {
            // Warmup iterations
            for (size_t i = 0; i < warmup_iters; ++i) {
                ch.send(send_buf.data(), msg_size);
                ch.recv(recv_buf.data(), msg_size);
            }
            
            // Timed iterations
            auto start = std::chrono::high_resolution_clock::now();
            
            for (size_t i = 0; i < iters; ++i) {
                ch.send(send_buf.data(), msg_size);
                ch.recv(recv_buf.data(), msg_size);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
            
            // Verify correctness
            if (send_buf != recv_buf) {
                overall_pass = false;
                std::cerr << "ERROR: Data mismatch at msg_size=" << msg_size << "\n";
            }
            
            // Calculate metrics (round-trip latency)
            double latency_usec = static_cast<double>(elapsed_us) / iters;
            double bandwidth_gbps = (msg_size * 8.0 * iters) / 
                                    (elapsed_us * 1000.0);  // Gbps
            
            rows.push_back({msg_size, bandwidth_gbps, latency_usec});
            
            std::cout << "msg_size=" << msg_size 
                      << " latency=" << latency_usec << "us"
                      << " bandwidth=" << bandwidth_gbps << "Gbps\n";
                      
        } catch (const std::exception& e) {
            overall_pass = false;
            std::cerr << "ERROR at msg_size=" << msg_size << ": " << e.what() << "\n";
        }
    }
    
    return {overall_pass, rows};
}


static void printJsonResult(bool overall_pass,
                            const std::vector<MetricRow>& rows) {
  // Output JSON
  std::cout << "{\n";
  std::cout << "  \"Correctness\": \"" << (overall_pass ? "PASS" : "FAIL") << "\",\n";
  std::cout << "  \"msg_size_unit\": \"bytes\",\n";
  std::cout << "  \"bandwidth_unit\": \"Gbps\",\n";
  std::cout << "  \"latency_unit\": \"usec\",\n";
  std::cout << "  \"metrics\": [\n";

  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::cout << "    {\"msg_size\": " << r.msg_size
              << ", \"bandwidth_gbps\": " << r.bandwidth_gbps
              << ", \"latency_usec\": " << r.latency_usec << "}";
    if (i < rows.size() - 1) std::cout << ",";
    std::cout << "\n";
  }
  std::cout << "  ]\n";
  std::cout << "}\n";
  
  // Also output text format for build_and_run.py regex parsing
  if (!rows.empty()) {
    const auto& last_row = rows.back();
    std::cout << "\nMessage size: " << last_row.msg_size << " bytes\n";
    std::cout << "Iterations: 100000\n";
    std::cout << "Bandwidth: " << last_row.bandwidth_gbps << " Gbit/s\n";
    std::cout << "Round-trip latency: " << last_row.latency_usec << " usec\n";
  }
  
  // Output PASSED/FAILED for regex matching
  std::cout << (overall_pass ? "PASSED" : "FAILED") << "\n";
}

int main(int argc, char** argv) {
    std::string device_name = "mlx5_0";
    std::string peer_ip = "";
    size_t msg_size = 4096;
    
    // Parse arguments
    parse_optional_args(argc, argv, msg_size);
    
    if (argc > 1 && argv[1][0] != '-') {
        device_name = argv[1];
    }
    if (argc > 2 && argv[2][0] != '-') {
        peer_ip = argv[2];  // empty string = server mode
    }

    try {
        RcChannel ch(device_name, 1, 1);
        ch.connect(peer_ip);

        // test different message sizes
        // 1KB, 4KB, 16KB, 64KB, 256KB, 1MB
        std::vector<size_t> msg_sizes = {
            1024, 4096, 16384, 65536, 262144, 1048576
        };
        constexpr size_t warmup = 1000;
        constexpr size_t iters = 100000;

        auto [overall_pass, rows] = runTest(ch, msg_sizes, warmup, iters);
        printJsonResult(overall_pass, rows);

        return overall_pass ? EXIT_SUCCESS : EXIT_FAILURE;

    }
    catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

}


// Minimal RC send/recv channel distilled from rc_pingpong.c
// Build: g++ -O2 -std=c++17 empty_rdma_recv_send.cpp -libverbs -o empty_rdma_recv_send
// two terminal windows
// on server side:
//   ./empty_rdma_recv_send ionic_0
// on client side:
//   ./empty_rdma_recv_send ionic_1 127.0.0.1