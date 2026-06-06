// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Unified FIFO Device-to-Host Communication Example
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

// ============================================================================
// Device Assertions
// ============================================================================

#if defined(MSCCLPP_DEVICE_COMPILE)

#if !defined(DEBUG_BUILD)
#define MSCCLPP_ASSERT_DEVICE(__cond, __msg)
#else

#if defined(MSCCLPP_DEVICE_HIP)
extern "C" __device__ void __assert_fail(char const* __assertion,
                                         char const* __file,
                                         unsigned int __line,
                                         char const* __function);
#else
extern "C" __host__ __device__ void __assert_fail(
    char const* __assertion, char const* __file, unsigned int __line,
    char const* __function) __THROW;
#endif

#define MSCCLPP_ASSERT_DEVICE(__cond, __msg)                         \
  do {                                                               \
    if (!(__cond)) {                                                 \
      __assert_fail(__msg, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    }                                                                \
  } while (0)

#endif  // !defined(DEBUG_BUILD)

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

// ============================================================================
// Polling Macros
// ============================================================================

#if defined(MSCCLPP_DEVICE_COMPILE)

#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt)                        \
  do {                                                                      \
    [[maybe_unused]] int64_t __spin_cnt = 0;                                \
    while (__cond) {                                                        \
      MSCCLPP_ASSERT_DEVICE(                                                \
          (__max_spin_cnt < 0 || __spin_cnt++ != __max_spin_cnt), #__cond); \
    }                                                                       \
  } while (0);

#endif

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

}  // namespace detail

// ============================================================================
// Trigger Types and Constants
// ============================================================================

using TriggerType = uint64_t;
constexpr TriggerType TriggerData = 0x1;
constexpr TriggerType TriggerFlag = 0x2;
constexpr TriggerType TriggerSync = 0x4;

constexpr unsigned int TriggerBitsSize = 32;
constexpr unsigned int TriggerBitsOffset = 32;
constexpr unsigned int TriggerBitsMemoryId = 9;
constexpr unsigned int TriggerBitsType = 3;
constexpr unsigned int TriggerBitsSemaphoreId = 10;
constexpr unsigned int TriggerBitsFifoReserved = 1;

// ============================================================================
// ProxyTrigger - Work element for the FIFO
// ============================================================================

union alignas(16) ProxyTrigger {
  struct {
    uint64_t fst;
    uint64_t snd;
  };
  struct {
    uint64_t size : TriggerBitsSize;
    uint64_t srcOffset : TriggerBitsOffset;
    uint64_t : (64 - TriggerBitsSize - TriggerBitsOffset);
    uint64_t dstOffset : TriggerBitsOffset;
    uint64_t srcMemoryId : TriggerBitsMemoryId;
    uint64_t dstMemoryId : TriggerBitsMemoryId;
    uint64_t type : TriggerBitsType;
    uint64_t semaphoreId : TriggerBitsSemaphoreId;
    uint64_t : (64 - TriggerBitsOffset - TriggerBitsMemoryId -
                TriggerBitsMemoryId - TriggerBitsType - TriggerBitsSemaphoreId -
                TriggerBitsFifoReserved);
    uint64_t reserved : TriggerBitsFifoReserved;
  } fields;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_INLINE ProxyTrigger() = default;

  MSCCLPP_DEVICE_INLINE ProxyTrigger(TriggerType type, uint32_t dstId,
                                     uint64_t dstOffset, uint32_t srcId,
                                     uint64_t srcOffset, uint64_t bytes,
                                     uint32_t semaphoreId) {
    MSCCLPP_ASSERT_DEVICE(type < (1ULL << TriggerBitsType),
                          "type is too large");
    MSCCLPP_ASSERT_DEVICE(dstId < (1ULL << TriggerBitsMemoryId),
                          "dstId is too large");
    MSCCLPP_ASSERT_DEVICE(dstOffset < (1ULL << TriggerBitsOffset),
                          "dstOffset is too large");
    MSCCLPP_ASSERT_DEVICE(srcId < (1ULL << TriggerBitsMemoryId),
                          "srcId is too large");
    MSCCLPP_ASSERT_DEVICE(srcOffset < (1ULL << TriggerBitsOffset),
                          "srcOffset is too large");
    MSCCLPP_ASSERT_DEVICE(bytes != 0, "bytes must not be zero");
    MSCCLPP_ASSERT_DEVICE(bytes < (1ULL << TriggerBitsSize),
                          "bytes is too large");
    MSCCLPP_ASSERT_DEVICE(semaphoreId < (1ULL << TriggerBitsSemaphoreId),
                          "semaphoreId is too large");
    constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
    constexpr uint64_t maskSemaphoreId = (1ULL << TriggerBitsSemaphoreId) - 1;
    fst =
        (((srcOffset & maskSrcOffset) << TriggerBitsSize) + (bytes & maskSize));
    snd = (((((((((semaphoreId & maskSemaphoreId) << TriggerBitsType) +
                 ((uint64_t)type & maskType))
                << TriggerBitsMemoryId) +
               (dstId & maskDstMemoryId))
              << TriggerBitsMemoryId) +
             (srcId & maskSrcMemoryId))
            << TriggerBitsOffset) +
           (dstOffset & maskDstOffset));
  }
#endif
};

// ============================================================================
// FifoDeviceHandle - Device-side FIFO access
// ============================================================================

struct FifoDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE uint64_t push(ProxyTrigger trigger,
                                      int64_t maxSpinCount = 1000000) {
     // TODO                                   
  }

  MSCCLPP_DEVICE_INLINE bool poll(uint64_t fifoHead) {
     // TODO      
  }

  MSCCLPP_DEVICE_INLINE void sync(
      uint64_t fifoHead, [[maybe_unused]] int64_t maxSpinCount = 1000000) {
    // TODO      
  }
#endif

  ProxyTrigger* triggers;
  uint64_t* head;
  uint64_t* tail;
  uint64_t* tailCache;
  int size;
};

// ============================================================================
// Fifo Class - Host-side FIFO management
// ============================================================================

constexpr size_t DEFAULT_FIFO_SIZE = 512;

class Fifo {
 public:
  Fifo(int size = DEFAULT_FIFO_SIZE) : size_(size) {
    // TODO      
  }

  ~Fifo() = default;

  ProxyTrigger poll() {
    // TODO      
  }

  void pop() {
    // TODO      
  }

  int size() const { return size_; }

  FifoDeviceHandle deviceHandle() const {
    FifoDeviceHandle handle;
    handle.triggers = triggers_.get();
    handle.head = head_.get();
    handle.tail = tail_.get();
    handle.tailCache = tailCache_.get();
    handle.size = size_;
    return handle;
  }

 private:
  detail::UniqueGpuHostPtr<ProxyTrigger> triggers_;
  detail::UniqueGpuPtr<uint64_t> head_;
  detail::UniqueGpuHostPtr<uint64_t> tail_;
  detail::UniqueGpuPtr<uint64_t> tailCache_;
  int size_;
};

}  // namespace mscclpp

// ============================================================================
// Test Kernel - GPU pushes work items to FIFO
// ============================================================================
__device__ __forceinline__ uint32_t get_smid() {
  uint32_t smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}
__global__ void testFifoPushKernel(mscclpp::FifoDeviceHandle fifo,
                                   int numPushes, uint64_t* gpu_timestamps) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < numPushes; i += stride) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = (uint64_t)(i + 1);

    uint64_t ts = clock64();   // GPU timestamp
    trigger.snd = ts;
    // printf("GPU Push: tid=%d, i=%d, ts=%llu\n", tid, i, ts);
    fifo.push(trigger);
    gpu_timestamps[i] = ts;  // optional: debug / validation
  }
}

inline uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + ts.tv_nsec;
}

// ============================================================================
// Host Proxy Thread - Consumes work items from FIFO
// ============================================================================

void hostProxyThread(mscclpp::Fifo& fifo,
                     int expectedItems,
                     std::vector<uint64_t>& latencies_ns,
                     std::atomic<int>& processedCount,
                     std::atomic<bool>& shouldStop,
                     double gpu_to_cpu_ns_scale,
                     uint64_t gpu_epoch_ns) {
  constexpr uint64_t flipMask = uint64_t{1} << 63;

  while (!shouldStop.load() && processedCount.load() < expectedItems) {
    mscclpp::ProxyTrigger trigger = fifo.poll();
    if (trigger.fst != 0) {
      uint64_t cpu_now = now_ns();

      int itemId = int(trigger.fst - 1);
      // push() XORs bit-63 as a validity flag – undo it to recover the
      // original GPU clock64() timestamp.
      uint64_t gpu_ts = trigger.snd ^ flipMask;

      // Map GPU clock64 tick to CPU CLOCK_MONOTONIC_RAW timeline
      uint64_t gpu_time_ns =
          gpu_epoch_ns + uint64_t(double(gpu_ts) * gpu_to_cpu_ns_scale);

      // Guard against calibration jitter: if cpu_now < gpu_time_ns the
      // subtraction would underflow, so clamp to 0.
      latencies_ns[itemId] =
          (cpu_now >= gpu_time_ns) ? (cpu_now - gpu_time_ns) : 0;
      fifo.pop();
      processedCount.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

__global__ void gpuClockSample(uint64_t* out) {
  if (threadIdx.x == 0) out[0] = clock64();
}


struct ClockCalibration {
  double gpu_tick_ns;    // ns per GPU clock tick
  uint64_t gpu_epoch_ns; // CPU time (ns) that corresponds to GPU clock == 0
};

ClockCalibration calibrateGpuClock(int warmupIters = 5, int calibIters = 20) {
  uint64_t* d_clock;
  MSCCLPP_CUDATHROW(cudaMalloc(&d_clock, sizeof(uint64_t)));

  // Warm up the GPU launch path
  for (int i = 0; i < warmupIters; i++) {
    gpuClockSample<<<1, 1>>>(d_clock);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  }

  uint64_t bestBracket = UINT64_MAX;
  uint64_t bestCpuMid = 0;
  uint64_t bestGpuClock = 0;

  for (int i = 0; i < calibIters; i++) {
    uint64_t cpuBefore = now_ns();
    gpuClockSample<<<1, 1>>>(d_clock);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    uint64_t cpuAfter = now_ns();

    uint64_t gpuClock;
    MSCCLPP_CUDATHROW(cudaMemcpy(&gpuClock, d_clock, sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost));

    uint64_t bracket = cpuAfter - cpuBefore;
    if (bracket < bestBracket) {
      bestBracket = bracket;
      bestCpuMid = (cpuBefore + cpuAfter) / 2;
      bestGpuClock = gpuClock;
    }
  }

  MSCCLPP_CUDATHROW(cudaFree(d_clock));

  // Get GPU clock rate from device properties
#if defined(MSCCLPP_DEVICE_HIP)
  hipDeviceProp_t prop;
  MSCCLPP_CUDATHROW(hipGetDeviceProperties(&prop, 0));
#else
  cudaDeviceProp prop;
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&prop, 0));
#endif
  // clockRate is in kHz => ticks per second = clockRate * 1000
  double gpu_tick_ns = 1e9 / (double(prop.clockRate) * 1000.0);

  // gpu_epoch_ns: the CPU time when GPU clock was 0
  // cpuMid ≈ gpu_epoch_ns + bestGpuClock * gpu_tick_ns
  uint64_t gpu_epoch_ns =
      bestCpuMid - uint64_t(double(bestGpuClock) * gpu_tick_ns);
  return {gpu_tick_ns, gpu_epoch_ns};
}

// ============================================================================
// Parse comma-separated integers from string, e.g. "32,64,128"
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
// Test result for one configuration
// ============================================================================
struct TestResult {
  int blockSize;
  int numPushes;
  int processed;
  double wall_us;
  double items_per_sec;
  double throughput_MBps;
  uint64_t lat_min_ns;
  uint64_t lat_avg_ns;
  uint64_t lat_p50_ns;
  uint64_t lat_p99_ns;
  uint64_t lat_max_ns;
  bool pass;
};

// ============================================================================
// Run one test configuration
// ============================================================================
TestResult runTest(int blockSize, int numPushes, const ClockCalibration& calib) {
  const int fifoSize = 512;
  mscclpp::Fifo fifo(fifoSize);
  mscclpp::FifoDeviceHandle deviceHandle = fifo.deviceHandle();

  int numBlocks = (numPushes + blockSize - 1) / blockSize;

  // Allocate GPU timestamp array
  uint64_t* d_gpu_ts;
  MSCCLPP_CUDATHROW(cudaMalloc(&d_gpu_ts, numPushes * sizeof(uint64_t)));
  MSCCLPP_CUDATHROW(cudaMemset(d_gpu_ts, 0, numPushes * sizeof(uint64_t)));

  // Per-item latency storage (filled by proxy thread)
  std::vector<uint64_t> latencies_ns(numPushes, 0);
  std::atomic<int> processedCount(0);
  std::atomic<bool> shouldStop(false);

  // Start host proxy thread
  std::thread proxyThread(hostProxyThread, std::ref(fifo), numPushes,
                          std::ref(latencies_ns), std::ref(processedCount),
                          std::ref(shouldStop), calib.gpu_tick_ns,
                          calib.gpu_epoch_ns);

  // Small delay to let proxy thread spin up
  std::this_thread::sleep_for(std::chrono::microseconds(500));

  // ---- Timing starts here ----
  uint64_t wallStart = now_ns();

  testFifoPushKernel<<<2, blockSize>>>(deviceHandle, numPushes,
                                                d_gpu_ts);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  // Wait for all items to be consumed
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (processedCount.load(std::memory_order_acquire) < numPushes) {
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "[Warning] Timeout: only " << processedCount.load()
                << "/" << numPushes << " items processed\n";
      break;
    }
    // Busy-wait with minimal sleep
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }

  uint64_t wallEnd = now_ns();
  // ---- Timing ends here ----

  shouldStop.store(true, std::memory_order_release);
  proxyThread.join();

  // Compute wall-clock throughput
  double wallElapsed_us = double(wallEnd - wallStart) / 1000.0;
  int processed = processedCount.load();
  double items_per_sec = double(processed) / (wallElapsed_us * 1e-6);
  double bytes_total = double(processed) * 16.0; // ProxyTrigger = 16 bytes
  double throughput_MBps = bytes_total / (wallElapsed_us * 1e-6) / (1024.0 * 1024.0);

  // Compute per-item latency statistics
  std::vector<uint64_t> valid_lat;
  valid_lat.reserve(processed);
  for (int i = 0; i < numPushes; i++) {
    if (latencies_ns[i] > 0) {
      valid_lat.push_back(latencies_ns[i]);
    }
  }
  std::sort(valid_lat.begin(), valid_lat.end());

  uint64_t lat_min = 0, lat_max = 0, lat_p50 = 0, lat_p99 = 0;
  double lat_avg = 0;
  if (!valid_lat.empty()) {
    lat_min = valid_lat.front();
    lat_max = valid_lat.back();
    lat_p50 = percentile(valid_lat, 50.0);
    lat_p99 = percentile(valid_lat, 99.0);
    uint64_t sum = 0;
    for (auto v : valid_lat) sum += v;
    lat_avg = double(sum) / double(valid_lat.size());
  }

  // Diagnostic output to stderr
  std::cerr << "  blockSize=" << blockSize << "  numPushes=" << numPushes
            << "  processed=" << processed << "/" << numPushes << "\n";
  std::cerr << "    Wall time:      " << wallElapsed_us << " us\n";
  std::cerr << "    Throughput:     " << items_per_sec << " items/s  ("
            << throughput_MBps << " MB/s)\n";
  std::cerr << "    Latency (ns):   min=" << lat_min << "  avg="
            << (uint64_t)lat_avg << "  p50=" << lat_p50 << "  p99=" << lat_p99
            << "  max=" << lat_max << "\n";
  std::cerr << "    Status:         "
            << (processed == numPushes ? "PASS" : "FAIL") << "\n";

  MSCCLPP_CUDATHROW(cudaFree(d_gpu_ts));

  return TestResult{
    blockSize, numPushes, processed,
    wallElapsed_us, items_per_sec, throughput_MBps,
    lat_min, (uint64_t)lat_avg, lat_p50, lat_p99, lat_max,
    processed == numPushes
  };
}

// ============================================================================
// Bench-trial aggregation
// ============================================================================
//
// Each (blockSize, numPushes) combination is too short for a single run to be
// stable on a busy host (kernel launch jitter, OS scheduling, GPU clock
// transitions can swing per-item latency by several ×). We therefore run
// 1 warmup trial (discarded) followed by kBenchTrials measured trials per
// combination, and aggregate by per-metric MEDIAN. Median is robust to a
// single bad sample, which is what we want for noise rejection.

static constexpr int kBenchTrials = 9;  // odd → unambiguous median index

// Pick the "median trial" by sorting the trials on a primary metric
// (lat_p50_ns) and returning the middle one whole. Reporting an entire
// trial keeps every metric internally consistent (e.g. lat_avg ≥ lat_min,
// lat_p99 ≥ lat_p50, items_per_sec consistent with wall_us). Cross-metric
// medians, taken independently, can produce impossible tuples on noisy
// hosts.
static TestResult aggregateTrials(const std::vector<TestResult>& trials) {
  if (trials.empty()) return TestResult{};
  std::vector<TestResult> sorted = trials;
  std::sort(sorted.begin(), sorted.end(),
            [](const TestResult& a, const TestResult& b) {
              return a.lat_p50_ns < b.lat_p50_ns;
            });
  TestResult median_trial = sorted[sorted.size() / 2];
  // pass flag: all-trials-passed only — one bad trial taints the combo.
  bool all_pass = true;
  for (const auto& t : trials) all_pass = all_pass && t.pass;
  median_trial.pass = all_pass;
  return median_trial;
}

// ============================================================================
// Main
// ============================================================================
//
// Usage:
//   ./fifo_test [-b threadblockSizes] [-n numPushes]
//
//   threadblockSizes  – comma-separated list, e.g. "32,64,128"   (default: 32)
//   numPushes   – comma-separated list, e.g. "10,100,1000" (default: 10,100)
//
// Every combination of (blockSize, numPushes) will be tested.

int main(int argc, char** argv) {
  std::vector<int> threadblockSizes = {256};
  std::vector<int> pushCounts = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

  // Simple argument parsing
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-b" || arg == "--block-sizes") && i + 1 < argc) {
      threadblockSizes = parseIntList(argv[++i]);
    } else if ((arg == "-n" || arg == "--num-pushes") && i + 1 < argc) {
      pushCounts = parseIntList(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cerr << "Usage: " << argv[0]
                << " [-b threadblockSizes] [-n numPushes]\n"
                << "  -b  comma-separated block sizes  (default: 256)\n"
                << "  -n  comma-separated push counts  (default: 32,64,...,65536)\n";
      return 0;
    }
  }

  std::vector<TestResult> allResults;
  bool allPass = true;

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

    // Calibrate GPU ↔ CPU clocks
    ClockCalibration calib = calibrateGpuClock();
    std::cerr << "\n";

    // Print test matrix to stderr
    std::cerr << "[Config] threadblockSizes: ";
    for (auto b : threadblockSizes) std::cerr << b << " ";
    std::cerr << "\n[Config] numPushes:  ";
    for (auto n : pushCounts) std::cerr << n << " ";
    std::cerr << "\n\n";

    // Run all combinations.
    // Per combo: 1 warmup trial (discarded) + kBenchTrials measured trials,
    // aggregated by per-metric median. Suppresses cold-cache and OS-jitter
    // outliers that otherwise make this short benchmark non-reproducible.
    for (int bs : threadblockSizes) {
      for (int np : pushCounts) {
        std::cerr << "----------------------------------------------\n";
        std::cerr << "[Trial] warmup (discarded)\n";
        (void)runTest(bs, np, calib);
        std::vector<TestResult> trials;
        trials.reserve(kBenchTrials);
        for (int t = 0; t < kBenchTrials; t++) {
          std::cerr << "[Trial] " << (t + 1) << "/" << kBenchTrials << "\n";
          trials.push_back(runTest(bs, np, calib));
        }
        TestResult r = aggregateTrials(trials);
        std::cerr << "[Aggregated median] wall=" << r.wall_us << " us"
                  << "  items/s=" << r.items_per_sec
                  << "  lat_p50=" << r.lat_p50_ns << " ns"
                  << "  lat_p99=" << r.lat_p99_ns << " ns\n";
        allResults.push_back(r);
        if (!r.pass) allPass = false;
        std::cerr << "\n";
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "[Error] " << e.what() << std::endl;
    // Output minimal JSON even on error
    std::cout << "{\"Correctness\": \"FAIL\"}" << std::endl;
    return 1;
  }

  // ── Output single JSON to stdout ──────────────────────────────────────
  std::cout << "{\n";
  std::cout << "  \"Correctness\": \"" << (allPass ? "PASS" : "FAIL") << "\",\n";
  std::cout << "  \"block_size_unit\": \"threads\",\n";
  std::cout << "  \"num_pushes_unit\": \"count\",\n";
  std::cout << "  \"wall_time_unit\": \"us\",\n";
  std::cout << "  \"items_per_sec_unit\": \"items/s\",\n";
  std::cout << "  \"throughput_unit\": \"MBps\",\n";
  std::cout << "  \"latency_unit\": \"ns\",\n";
  std::cout << "  \"metrics\": [\n";

  for (size_t i = 0; i < allResults.size(); i++) {
    const auto& r = allResults[i];
    std::cout << "    {\"block_size\": " << r.blockSize
              << ", \"num_pushes\": " << r.numPushes
              << ", \"processed\": " << r.processed
              << ", \"wall_us\": " << r.wall_us
              << ", \"items_per_sec\": " << r.items_per_sec
              << ", \"throughput_MBps\": " << r.throughput_MBps
              << ", \"lat_min_ns\": " << r.lat_min_ns
              << ", \"lat_avg_ns\": " << r.lat_avg_ns
              << ", \"lat_p50_ns\": " << r.lat_p50_ns
              << ", \"lat_p99_ns\": " << r.lat_p99_ns
              << ", \"lat_max_ns\": " << r.lat_max_ns
              << ", \"pass\": " << (r.pass ? "true" : "false")
              << "}" << (i + 1 < allResults.size() ? ",\n" : "\n");
  }

  std::cout << "  ]\n";
  std::cout << "}" << std::endl;
  return 0;
}
