// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Unified FIFO Host-to-Device Communication Example
// This file combines all FIFO implementation and test code into a single file.
// Supports both CUDA (NVIDIA) and HIP (AMD) platforms.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <algorithm>
#include <ctime>
#include <iterator>
// gdrcopy was the original mapping mechanism for the device-side `tail`
// counter. We now use cudaHostAllocMapped (see Impl::tail below) so the
// build no longer needs gdrcopy.
// ============================================================================
// GPU Platform Abstraction
// ============================================================================

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define MSCCLPP_DEVICE_HIP
// Map CUDA API to HIP API
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaMalloc hipMalloc
#define cudaMemset hipMemset
#define cudaFree hipFree
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaGetLastError hipGetLastError
#define cudaErrorContextIsDestroyed hipErrorContextIsDestroyed
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaLaunchKernel hipLaunchKernel
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#else
#include <cuda.h>
#include <cuda_runtime.h>
#define MSCCLPP_DEVICE_CUDA
#endif

// ============================================================================
// Device Compilation Macros
// ============================================================================

#if (defined(__NVCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE __forceinline__
#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#else
#define MSCCLPP_HOST_COMPILE
#define MSCCLPP_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline
#endif

// ============================================================================
// CUDA Atomic includes (for NVIDIA only)
// ============================================================================
#if defined(MSCCLPP_DEVICE_CUDA) && defined(__NVCC__)
#include <cuda/atomic>
#endif

namespace mscclpp {

// ============================================================================
// Error Handling
// ============================================================================

class BaseError : public std::runtime_error {
 public:
  BaseError(std::string const& message, int errorCode)
      : std::runtime_error(""), message_(message), errorCode_(errorCode) {}

  explicit BaseError(int errorCode)
      : std::runtime_error(""), errorCode_(errorCode) {}

  virtual ~BaseError() = default;

  int getErrorCode() const { return errorCode_; }

  char const* what() const noexcept override { return message_.c_str(); }

 protected:
  std::string message_;
  int errorCode_;
};

class CudaError : public BaseError {
 public:
  CudaError(std::string const& message, int errorCode) : BaseError(errorCode) {
    message_ = message + " (GPU failure: " +
               cudaGetErrorString(static_cast<cudaError_t>(errorCode)) + ")";
  }
  virtual ~CudaError() = default;
};

// ============================================================================
// GPU Error Checking Macros
// ============================================================================

#define MSCCLPP_CUDATHROW(cmd)                                              \
  do {                                                                      \
    cudaError_t err = cmd;                                                  \
    if (err != cudaSuccess) {                                               \
      throw ::mscclpp::CudaError(std::string("Call to " #cmd " failed. ") + \
                                     __FILE__ + ":" +                       \
                                     std::to_string(__LINE__),              \
                                 err);                                      \
    }                                                                       \
  } while (false)

// ============================================================================
// NUMA helper (simplified - just read numa node, no mempolicy)
// ============================================================================

static inline std::string const getBusId(int deviceId) {
  char busIdChar[] = "00000000:00:00.0";
  MSCCLPP_CUDATHROW(
      cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

inline int getDeviceNumaNode(int deviceId) {
  std::string busId = getBusId(deviceId);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/numa_node";
  std::ifstream file(file_str);
  int numaNode = -1;
  if (file.is_open()) {
    if (!(file >> numaNode)) {
      return -1;
    }
  }
  return numaNode;
}

// ============================================================================
// Atomic Operations
// ============================================================================

#if defined(MSCCLPP_DEVICE_CUDA) && defined(__NVCC__)

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;
constexpr cuda::memory_order memoryOrderRelease = cuda::memory_order_release;
constexpr cuda::memory_order memoryOrderAcqRel = cuda::memory_order_acq_rel;
constexpr cuda::memory_order memoryOrderSeqCst = cuda::memory_order_seq_cst;

constexpr cuda::thread_scope scopeSystem = cuda::thread_scope_system;
constexpr cuda::thread_scope scopeDevice = cuda::thread_scope_device;

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(T* ptr,
                                        cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, Scope>{*ptr}.load(memoryOrder);
}

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, T const& val,
                                            cuda::memory_order memoryOrder) {
  cuda::atomic_ref<T, Scope>{*ptr}.store(val, memoryOrder);
}

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, T const& val,
                                            cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, Scope>{*ptr}.fetch_add(val, memoryOrder);
}

#else  // HIP or Host compilation

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

constexpr auto scopeSystem = 0;
constexpr auto scopeDevice = 0;

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(T const* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, T const& val,
                                            int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, T const& val,
                                            int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#endif

#define CUCHK(x)                                                     \
  do {                                                               \
    CUresult r = (x);                                                \
    if (r != CUDA_SUCCESS) {                                         \
      const char* s;                                                 \
      cuGetErrorString(r, &s);                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
              __FILE__, __LINE__, s);                                \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

#define CUDA_RTCHK(x)                                                \
  do {                                                               \
    cudaError_t e = (x);                                             \
    if (e != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA RT error %s:%d: %s\n",                   \
              __FILE__, __LINE__, cudaGetErrorString(e));            \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

// ============================================================================
// GPU Memory Management
// ============================================================================

namespace detail {

static inline bool isCudaTeardownError(cudaError_t err) {
#if defined(MSCCLPP_DEVICE_HIP)
  return err == cudaErrorContextIsDestroyed || err == cudaErrorInvalidDevice;
#else
  return err == cudaErrorCudartUnloading ||
         err == cudaErrorContextIsDestroyed ||
         err == cudaErrorInitializationError || err == cudaErrorInvalidDevice ||
         err == cudaErrorLaunchFailure;
#endif
}

inline void* gpuCalloc(size_t bytes) {
  void* ptr;
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, bytes));
  MSCCLPP_CUDATHROW(cudaMemset(ptr, 0, bytes));
  return ptr;
}

inline void* gpuCallocHost(size_t bytes, unsigned int flags) {
  void* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, bytes, flags));
  ::memset(ptr, 0, bytes);
  return ptr;
}

inline void _gpuFree(void* ptr) {
  cudaError_t err = cudaFree(ptr);
  if (!isCudaTeardownError(err) && err != cudaSuccess) {
    throw ::mscclpp::CudaError(std::string("Call to cudaFree failed. ") +
                                   __FILE__ + ":" + std::to_string(__LINE__),
                               err);
  } else if (isCudaTeardownError(err)) {
    (void)cudaGetLastError();
  }
}

inline void _gpuFreeHost(void* ptr) {
  cudaError_t err = cudaFreeHost(ptr);
  if (!isCudaTeardownError(err) && err != cudaSuccess) {
    throw ::mscclpp::CudaError(std::string("Call to cudaFreeHost failed. ") +
                                   __FILE__ + ":" + std::to_string(__LINE__),
                               err);
  } else if (isCudaTeardownError(err)) {
    (void)cudaGetLastError();
  }
}

template <class T = void>
struct GpuDeleter {
  void operator()(void* ptr) { _gpuFree(ptr); }
};

template <class T = void>
struct GpuHostDeleter {
  void operator()(void* ptr) { _gpuFreeHost(ptr); }
};

template <class T>
using UniqueGpuPtr = std::unique_ptr<T, detail::GpuDeleter<T>>;

template <class T>
using UniqueGpuHostPtr = std::unique_ptr<T, detail::GpuHostDeleter<T>>;

template <class T, class Deleter, class Memory, typename Alloc,
          typename... Args>
Memory safeAlloc(Alloc alloc, size_t nelems, Args&&... args) {
  T* ptr = nullptr;
  try {
    ptr = reinterpret_cast<T*>(
        alloc(nelems * sizeof(T), std::forward<Args>(args)...));
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

template <class T>
auto gpuCallocUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuDeleter<T>, UniqueGpuPtr<T>>(
      detail::gpuCalloc, nelems);
}

template <class T>
auto gpuCallocHostUnique(size_t nelems = 1,
                         unsigned int flags = cudaHostAllocMapped) {
  return detail::safeAlloc<T, detail::GpuHostDeleter<T>, UniqueGpuHostPtr<T>>(
      detail::gpuCallocHost, nelems, flags);
}

// gdrcopy is intentionally not used: a single uint64_t `tail` counter is
// allocated as host-pinned mapped memory (cudaHostAllocMapped) so the GPU
// reaches it via UVA and the CPU just reads the same pointer. This gives
// the same "GPU-writes / CPU-reads" semantics gdrcopy was used for, without
// the gdrdrv kernel-module dependency.
inline uint64_t* getGdrHostPtr(detail::UniqueGpuHostPtr<uint64_t> const& p) {
  return p.get();
}



}  // namespace detail

// ============================================================================
// C2DDeviceHandle - Device-side handle for CPU-to-GPU FIFO
// ============================================================================

constexpr size_t DEFAULT_FIFO_SIZE = 512;

template <typename T>
struct C2DDeviceHandle {
  T* buffer;       // T FIFO on device
  uint64_t* head;  // host-pinned, updated by CPU
  uint64_t* tail;  // device, atomically consumed by GPU
  int size;        // Fifo Size

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Try to get a pointer to the next unconsumed task.
  /// @return Pointer to task if available, nullptr otherwise.
  MSCCLPP_DEVICE_INLINE T* poll() {
    // TODO
  }

  /// Consume the task at tail (advance tail by 1).
  /// Only call after poll() returns non-null.
  MSCCLPP_DEVICE_INLINE void pop() {
    // TODO
  }
#endif  // MSCCLPP_DEVICE_COMPILE
};

// ============================================================================
// CpuToGpuFifo - Host-side FIFO management for CPU-to-GPU direction
// ============================================================================

template <typename T>
class CpuToGpuFifo {
 public:
  explicit CpuToGpuFifo(int size = DEFAULT_FIFO_SIZE)
      : pimpl_(std::make_unique<Impl>(size)) {
    // TODO
  }

  ~CpuToGpuFifo() = default;

  /// Push a single task from CPU, return task id
  uint64_t push(const T& task) {
    // TODO
  }

  /// Push a range of tasks [first, last) from CPU.
  template <typename InputIt>
  uint64_t push(InputIt first, InputIt last) {
    // TODO
  }

  uint64_t head() const {
    // TODO
  }

  /// Read current tail value (how many items GPU has consumed).
  /// Uses host-pinned tail pointer which is accessible from host.
  uint64_t currentId() const {
    // TODO
  }

  /// Wait until GPU has consumed up to (and including) taskId.
  void sync(uint64_t taskId) const {
    // TODO
  }

  /// Get device handle for GPU kernels.
  C2DDeviceHandle<T> deviceHandle() const {
    C2DDeviceHandle<T> h;
    h.buffer = pimpl_->buffer.get();
    h.head = pimpl_->head.get();
    h.tail = pimpl_->tail.get();
    h.size = pimpl_->size;
    return h;
  }

  /// Get FIFO capacity (number of entries).
  int size() const { return pimpl_->size; }

 private:
  struct Impl {
    detail::UniqueGpuPtr<T> buffer;            // device memory for FIFO ring
    detail::UniqueGpuHostPtr<uint64_t> head;   // host-pinned, written by CPU
    detail::UniqueGpuHostPtr<uint64_t> tail;  // host-pinned mapped: GPU writes via UVA, CPU reads directly
    int const size;
    cudaStream_t h2d_stream;

    Impl(int size)
        : buffer(detail::gpuCallocUnique<T>(size)),
          head(detail::gpuCallocHostUnique<uint64_t>()),
          tail(detail::gpuCallocHostUnique<uint64_t>()),
          size(size),
          h2d_stream(nullptr) {
              MSCCLPP_CUDATHROW(
                  cudaStreamCreateWithFlags(&h2d_stream,
                                            cudaStreamNonBlocking));
          }
    ~Impl() {
      if (h2d_stream) {
        cudaStreamDestroy(h2d_stream);
      }
    }
  };
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

// ============================================================================
// Simple task type for testing
// ============================================================================

struct alignas(8) SimpleTask {
  uint64_t value;
};

// ============================================================================
// Test Kernel - GPU polls and consumes tasks from FIFO
// ============================================================================

__global__ void testFifoPopKernel(mscclpp::C2DDeviceHandle<SimpleTask> fifo,
                                  int expectedItems,
                                  uint64_t* results,
                                  int* processedCount) {
  // Only one thread handles the FIFO polling to avoid races
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  int count = 0;
  int spins = 0;
  constexpr int MAX_SPINS = 100000000;  // safety limit

  while (count < expectedItems && spins < MAX_SPINS) {
    SimpleTask* task = fifo.poll();
    if (task != nullptr) {
      // printf("GPU consumed task with value %lu at count %d\n", task->value, count);
      results[count] = task->value;
      fifo.pop();
      count++;
      spins = 0;  // reset spin counter on progress
    } else {
      spins++;
    }
  }

  *processedCount = count;
}

// ============================================================================
// Timing helper
// ============================================================================

inline uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + ts.tv_nsec;
}

// ============================================================================
// Parse comma-separated integers from string
// ============================================================================
std::vector<int> parseIntList(const std::string& s) {
  std::vector<int> result;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    if (end == std::string::npos) end = s.size();
    result.push_back(std::stoi(s.substr(start, end - start)));
    start = end + 1;
  }
  return result;
}

// ============================================================================
// Percentile helper (on sorted vector)
// ============================================================================
uint64_t percentile(const std::vector<uint64_t>& sorted, double p) {
  if (sorted.empty()) return 0;
  size_t idx = size_t(p / 100.0 * double(sorted.size() - 1));
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
}

// ============================================================================
// Run one test configuration
// ============================================================================

// Accumulator for per-run metrics so main() can emit a single aggregate JSON.
struct PerRunMetrics {
  int numPushes;
  int processed;
  int correct;
  double wall_us;
  double latency_us;
  double items_per_sec;
  double throughput_MBps;
  bool pass;
};
static std::vector<PerRunMetrics> g_metrics;

void runTest(int numPushes) {
  const int fifoSize = 512;
  mscclpp::CpuToGpuFifo<SimpleTask> fifo(fifoSize);
  auto deviceHandle = fifo.deviceHandle();

  // Allocate GPU-side result array and processed count
  uint64_t* d_results;
  int* d_processedCount;
  MSCCLPP_CUDATHROW(cudaMalloc(&d_results, numPushes * sizeof(uint64_t)));
  MSCCLPP_CUDATHROW(cudaMemset(d_results, 0, numPushes * sizeof(uint64_t)));
  MSCCLPP_CUDATHROW(cudaMalloc(&d_processedCount, sizeof(int)));
  MSCCLPP_CUDATHROW(cudaMemset(d_processedCount, 0, sizeof(int)));

  // Launch GPU consumer kernel (single thread for simplicity)
  testFifoPopKernel<<<1, 1>>>(deviceHandle, numPushes, d_results,
                               d_processedCount);

  // Small delay to let kernel start spinning
  std::this_thread::sleep_for(std::chrono::microseconds(500));

  // ---- Timing starts here ----
  uint64_t wallStart = now_ns();

  // CPU pushes tasks into FIFO
  for (int i = 0; i < numPushes; i++) {
    SimpleTask task;
    task.value = uint64_t(i + 1);
    fifo.push(task);
    // printf("CPU pushed task with value %lu at iteration %d\n", task.value, i);
  }

  // Wait for GPU kernel to finish
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  uint64_t wallEnd = now_ns();
  // ---- Timing ends here ----

  // Copy results back
  int processedCount = 0;
  MSCCLPP_CUDATHROW(cudaMemcpy(&processedCount, d_processedCount, sizeof(int),
                                cudaMemcpyDeviceToHost));

  std::vector<uint64_t> results(numPushes, 0);
  MSCCLPP_CUDATHROW(cudaMemcpy(results.data(), d_results,
                                numPushes * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost));

  // Verify correctness: each result[i] should be i+1
  int correctCount = 0;
  for (int i = 0; i < processedCount; i++) {
    if (results[i] == uint64_t(i + 1)) {
      correctCount++;
    }
  }

  // Compute wall-clock throughput
  double wallElapsed_us = double(wallEnd - wallStart) / 1000.0;
  double items_per_sec = double(processedCount) / (wallElapsed_us * 1e-6);
  double bytes_total = double(processedCount) * sizeof(SimpleTask);
  double throughput_MBps =
      bytes_total / (wallElapsed_us * 1e-6) / (1024.0 * 1024.0);
  double latency_us = (processedCount > 0) ? (wallElapsed_us / double(processedCount)) : 0.0;

  bool pass = (processedCount == numPushes) && (correctCount == numPushes);

  // Human-readable progress -> stderr so stdout stays clean for the
  // single aggregate JSON object printed at the end of main().
  std::cerr << "  numPushes=" << numPushes
            << "  processed=" << processedCount << "/" << numPushes
            << "  correct=" << correctCount << "/" << processedCount << "\n";
  std::cerr << "    Wall time:      " << wallElapsed_us << " us\n";
  std::cerr << "    Latency:        " << latency_us << " us/item\n";
  std::cerr << "    Throughput:     " << items_per_sec << " items/s  ("
            << throughput_MBps << " MB/s)\n";
  std::cerr << "    Status:         " << (pass ? "PASS" : "FAIL") << "\n";

  g_metrics.push_back({numPushes, processedCount, correctCount,
                       wallElapsed_us, latency_us, items_per_sec,
                       throughput_MBps, pass});

  MSCCLPP_CUDATHROW(cudaFree(d_results));
  MSCCLPP_CUDATHROW(cudaFree(d_processedCount));
}

// ============================================================================
// Main
// ============================================================================
//
// Usage:
//   ./h2dfifo_test [-n numPushes]
//
//   numPushes – comma-separated list, e.g. "10,100,1000" (default: 32,64,...,512)

int main(int argc, char** argv) {
  std::vector<int> pushCounts = {32, 64, 128, 256, 512};

  // Simple argument parsing
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--num-pushes") && i + 1 < argc) {
      pushCounts = parseIntList(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [-n numPushes]\n"
                << "  -n  comma-separated push counts  (default: 32,64,128,256,512)\n";
      return 0;
    }
  }

  std::cerr << "==============================================\n";
  std::cerr << "FIFO Host-to-Device Communication Benchmark\n";
  std::cerr << "==============================================\n\n";

  try {
    int deviceId = 0;
    MSCCLPP_CUDATHROW(cudaSetDevice(deviceId));
#if defined(MSCCLPP_DEVICE_HIP)
    hipDeviceProp_t devProp;
    MSCCLPP_CUDATHROW(hipGetDeviceProperties(&devProp, deviceId));
#else
    cudaDeviceProp devProp;
    MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&devProp, deviceId));
#endif
    std::cerr << "[Info] GPU: " << devProp.name << "\n\n";

    std::cerr << "[Config] numPushes: ";
    for (auto n : pushCounts) std::cerr << n << " ";
    std::cerr << "\n\n";

    // Run all configurations
    for (int np : pushCounts) {
      std::cerr << "----------------------------------------------\n";
      runTest(np);
      std::cerr << "\n";
    }

  } catch (const std::exception& e) {
    std::cerr << "[Error] " << e.what() << std::endl;
    return 1;
  }

  std::cerr << "==============================================\n";
  std::cerr << "All tests completed.\n";
  std::cerr << "==============================================\n";

  // Emit exactly one JSON object on stdout, aggregating all per-run metrics.
  bool all_pass = !g_metrics.empty();
  for (const auto& m : g_metrics) all_pass = all_pass && m.pass;

  std::cout << "{\n";
  std::cout << "  \"Correctness\": \"" << (all_pass ? "PASS" : "FAIL") << "\",\n";
  std::cout << "  \"data_size_unit\": \"items\",\n";
  std::cout << "  \"throughput_unit\": \"MBps\",\n";
  std::cout << "  \"latency_unit\": \"us\",\n";
  std::cout << "  \"metrics\": [\n";
  for (size_t i = 0; i < g_metrics.size(); ++i) {
    const auto& m = g_metrics[i];
    std::cout << "    {\"data_size\": " << m.numPushes
              << ", \"processed\": " << m.processed
              << ", \"correct\": " << m.correct
              << ", \"wall_us\": " << m.wall_us
              << ", \"latency_avg\": " << m.latency_us
              << ", \"items_per_sec\": " << m.items_per_sec
              << ", \"throughput_avg\": " << m.throughput_MBps
              << ", \"pass\": " << (m.pass ? "true" : "false") << "}"
              << (i + 1 == g_metrics.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n";
  std::cout << "}\n";
  return 0;
}
