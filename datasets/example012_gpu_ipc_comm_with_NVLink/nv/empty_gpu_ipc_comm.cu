// GPU Inter-Process Communication (IPC) benchmark using CUDA IPC memory handles.
// A sender process allocates GPU memory, fills it with test data, and exports an
// IPC memory handle plus an IPC event handle over a Unix domain socket. A receiver
// process imports both handles, waits for the sender's event, then performs a
// device-to-device copy (NVLink when two GPUs are present) from the shared remote
// memory into a local GPU buffer. The example measures latency and throughput of
// the D2D IPC read across multiple data sizes.

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(_e));                                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ── Unix socket communication class ─────────────────────────────────────────

static void die(const char* what) {
  std::perror(what);
  std::exit(EXIT_FAILURE);
}

class UnixSocketComm {
 public:
  UnixSocketComm() = default;
  ~UnixSocketComm() { close(); }

  // Server: create AF_UNIX socket, bind to path, and listen
  void listen(const char* path) {
    // TODO: Create an AF_UNIX SOCK_STREAM socket, unlink the path,
    // bind to it, and listen with backlog 1.
    // Store the listening fd in listenFd_ and the path in sockPath_.
  }

  // Server: accept a client connection
  void accept() {
    // TODO: Accept a connection on listenFd_.
    // Store the connected fd in connFd_.
  }

  // Client: connect to a listening server at path
  void connect(const char* path) {
    // TODO: Create an AF_UNIX SOCK_STREAM socket and connect to the
    // given path. Store the connected fd in connFd_.
  }

  // Send exactly n bytes from buf
  void sendAll(const void* buf, size_t n) {
    // TODO: Send exactly n bytes from buf over connFd_.
    // Loop calling ::send(), handle EINTR, call die("send") on error.
  }

  // Receive exactly n bytes into buf
  void recvAll(void* buf, size_t n) {
    // TODO: Receive exactly n bytes into buf from connFd_.
    // Loop calling ::recv(), handle EINTR, call die("recv") on error.
  }

  // Close all fds and unlink the socket file (server side only)
  void close() {
    if (connFd_ >= 0) {
      ::close(connFd_);
      connFd_ = -1;
    }
    if (listenFd_ >= 0) {
      ::close(listenFd_);
      listenFd_ = -1;
    }
    if (!sockPath_.empty()) {
      ::unlink(sockPath_.c_str());
      sockPath_.clear();
    }
  }

 private:
  int listenFd_ = -1;
  int connFd_ = -1;
  std::string sockPath_;
};

// ── IPC handles exchanged between processes ─────────────────────────────────

struct IpcHandles {
  cudaIpcMemHandle_t mem;
  cudaIpcEventHandle_t evt;
};

// ── CUDA IPC Communication class ────────────────────────────────────────────

class CudaIpcComm {
 public:
  CudaIpcComm(int device) : device_(device) {
    CUDA_CHECK(cudaSetDevice(device_));
  }

  ~CudaIpcComm() { cleanup(); }

  // Sender: allocate GPU memory
  void allocate(size_t bytes) {
    cleanup();
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaMalloc(&devPtr_, bytes));
    allocatedBytes_ = bytes;
  }

  // Sender: fill GPU memory with a float test pattern
  void initialize(size_t bytes) {
    // TODO: Initialize the first 'bytes' of GPU memory with a test pattern.
    // Fill a host vector of floats where element i = float(i % 1024),
    // then cudaMemcpy it to devPtr_ (HostToDevice).
  }

  // Sender: export IPC memory handle and create+export IPC event handle
  IpcHandles exportHandles() {
    // TODO: Export CUDA IPC handles for the allocated GPU memory and an
    // interprocess event.
    // 1. Use cudaIpcGetMemHandle to export the memory handle from devPtr_.
    // 2. Create an event with cudaEventInterprocess | cudaEventDisableTiming.
    // 3. Use cudaIpcGetEventHandle to export the event handle.
    // Store the event in ipcEvent_ and return both handles in IpcHandles.
    return {};
  }

  // Sender: record event to signal that data has been written
  void recordEvent() {
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaEventRecord(ipcEvent_, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Receiver: import IPC memory handle to access remote GPU memory
  void* importMemHandle(const cudaIpcMemHandle_t& handle) {
    // TODO: Open the IPC memory handle to get a device pointer to the
    // sender's GPU memory. Use cudaIpcOpenMemHandle with
    // cudaIpcMemLazyEnablePeerAccess. Return the device pointer.
    return nullptr;
  }

  // Receiver: import IPC event handle
  cudaEvent_t importEventHandle(const cudaIpcEventHandle_t& handle) {
    // TODO: Open the IPC event handle to get a local event reference.
    // Use cudaIpcOpenEventHandle. Return the event.
    return nullptr;
  }

  // Receiver: wait for the sender's event to complete
  void waitForSender(cudaEvent_t remoteEvent) {
    // TODO: Create a CUDA stream, call cudaStreamWaitEvent to wait for the
    // remote event, synchronize the stream, then destroy it.
  }

  // Receiver: verify that the D2D-copied data matches the expected pattern
  bool verify(const void* ptr, size_t bytes) {
    CUDA_CHECK(cudaSetDevice(device_));
    const size_t n = bytes / sizeof(float);
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
      if (h[i] != static_cast<float>(i % 1024)) return false;
    }
    return true;
  }

  static void closeMemHandle(void* ptr) {
    CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
  }

  static void closeEventHandle(cudaEvent_t evt) {
    CUDA_CHECK(cudaEventDestroy(evt));
  }

  int device() const { return device_; }
  void* devicePtr() const { return devPtr_; }

 private:
  int device_;
  void* devPtr_ = nullptr;
  size_t allocatedBytes_ = 0;
  cudaEvent_t ipcEvent_ = nullptr;

  void cleanup() {
    if (device_ >= 0) CUDA_CHECK(cudaSetDevice(device_));
    if (ipcEvent_) {
      CUDA_CHECK(cudaEventDestroy(ipcEvent_));
      ipcEvent_ = nullptr;
    }
    if (devPtr_) {
      CUDA_CHECK(cudaFree(devPtr_));
      devPtr_ = nullptr;
    }
    allocatedBytes_ = 0;
  }
};

// ── Test harness (correctness + performance benchmark) ──────────────────────

struct MetricRow {
  int data_size_mb;
  double throughput_avg_gbps;
  double latency_avg_us;
  bool pass;
};

static std::vector<int> default_sizes_mb() { return {1, 4, 16, 64, 256}; }

static const char* arg_str(int argc, char** argv, const char* key,
                           const char* defv) {
  for (int i = 1; i + 1 < argc; i++)
    if (std::string(argv[i]) == key) return argv[i + 1];
  return defv;
}

static int arg_int(int argc, char** argv, const char* key, int defv) {
  for (int i = 1; i + 1 < argc; i++)
    if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
  return defv;
}

// ── Sender process ──────────────────────────────────────────────────────────
//
// Protocol (per data size):
//   sender: initialize → recordEvent → send data_size via socket
//   receiver: recv data_size → waitEvent → benchmark D2D copy → send ACK
// Termination: sender sends data_size = 0.

static int runSender(const char* sock, int dev) {
  // TODO: Implement the sender process.
  // 1. Allocate max-sized GPU buffer via CudaIpcComm, export IPC handles.
  // 2. Create UnixSocketComm, listen on sock, accept connection.
  // 3. Send IpcHandles to receiver via socket.sendAll().
  // 4. For each size in default_sizes_mb():
  //    a. initialize(bytes), recordEvent()
  //    b. Send data_size (uint64_t) via socket.sendAll(), wait for 1-byte ACK
  //       via socket.recvAll().
  // 5. Send data_size = 0 as termination signal.
  // 6. Wait for final ACK from receiver, then socket.close().
  return 0;
}

// ── Receiver process ────────────────────────────────────────────────────────

static int runReceiver(const char* sock, int dev) {
  const int warmup_iters = 5;
  const int bench_iters = 20;

  std::vector<MetricRow> rows;
  bool overall_pass = true;

  // TODO: Implement the receiver process.
  // 1. Create UnixSocketComm, connect to sender via socket.connect(sock).
  // 2. Receive IpcHandles via socket.recvAll().
  // 3. Import memory handle and event handle via CudaIpcComm.
  // 4. Allocate a local GPU buffer for D2D copy.
  // 5. Loop: recv data_size (uint64_t) via socket.recvAll(); break if 0.
  //    a. waitForSender(remoteEvent)
  //    b. Warmup: cudaMemcpy(localPtr, remotePtr, bytes, D2D) x warmup_iters
  //    c. Timed: cudaMemcpy D2D x bench_iters, measure with cudaEvent timing
  //    d. Verify correctness via comm.verify(localPtr, bytes); update overall_pass
  //    e. Append MetricRow{data_size_mb, throughput_avg_gbps, latency_avg_us,
  //       pass} into `rows`
  //    f. Send 1-byte ACK via socket.sendAll()
  // 6. Close IPC handles (closeEventHandle, closeMemHandle), free local buffer.
  // 7. Send final ACK, socket.close().

  // Print JSON result — must use exactly these keys / units so the
  // framework's comparator (see build_and_run.py / perf_verdict) matches.
  std::cout << "{\n";
  std::cout << "  \"Correctness\": \""
            << (overall_pass ? "PASS" : "FAIL") << "\",\n";
  std::cout << "  \"data_size_unit\": \"MB\",\n";
  std::cout << "  \"throughput_unit\": \"Gbps\",\n";
  std::cout << "  \"latency_unit\": \"us\",\n";
  std::cout << "  \"metrics\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::cout << "    {\"data_size\": " << r.data_size_mb
              << ", \"throughput_avg\": " << r.throughput_avg_gbps
              << ", \"latency_avg\": " << r.latency_avg_us << "}";
    if (i + 1 != rows.size()) std::cout << ",";
    std::cout << "\n";
  }
  std::cout << "  ]\n";
  std::cout << "}\n";

  return overall_pass ? 0 : 1;
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  const char* role = arg_str(argc, argv, "--role", "sender");
  const char* sock = arg_str(argc, argv, "--sock", "/tmp/cuda_ipc.sock");
  int dev = arg_int(argc, argv, "--dev", 0);

  if (std::string(role) == "sender") return runSender(sock, dev);
  if (std::string(role) == "receiver") return runReceiver(sock, dev);

  std::fprintf(stderr, "use --role sender|receiver\n");
  return 1;
}
