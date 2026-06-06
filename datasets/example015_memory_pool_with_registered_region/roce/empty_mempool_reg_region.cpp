/*
 * GPU memory pool with a single RDMA memory registration.
 * Avoids calling ibv_reg_mr() in the hot path by pre-registering
 * one large buffer. Falls back to pinned host memory when
 * GPUDirect RDMA is unavailable.
*/
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <stdexcept>

// CUDA error checker
static void cc(cudaError_t e, const char* call, int line) {
    if (e == cudaSuccess) return;
    fprintf(stderr, "CUDA error at line %d (%s): %s\n",
            line, call, cudaGetErrorString(e));
    exit(1);
}
#define CC(expr) cc((expr), #expr, __LINE__)

/*  GPU memory pool with a single RDMA memory registration
 *  Avoids calling ibv_reg_mr() in the hot path by pre-registering
 *  one large buffer. Falls back to pinned host memory when
 *  GPUDirect RDMA is unavailable. 
 */
class MemoryPool {
public:
    MemoryPool(size_t bytes) : size_(bytes) {
        // TODO: Open the first RDMA device with ibv_get_device_list / ibv_open_device

        // TODO: Allocate a protection domain

        // TODO: Allocate GPU memory with cudaMalloc and try to register it with ibv_reg_mr

        // TODO: If ibv_reg_mr on GPU memory fails, fall back

        // TODO: Initialize stack pointers for slice allocation

    }

    ~MemoryPool() {
        // TODO: Cleanup resources in reverse order of creation
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // Return a pointer into the registered region at [offset, offset+len).
    // The pool's lkey/rkey are valid for this pointer.
    void* mr_slice(size_t offset, size_t len) const {
        if (offset + len > size_) {
            fprintf(stderr, "mr_slice: offset %zu + len %zu exceeds pool size %zu\n",
                    offset, len, size_);
            exit(1);
        }
        return (char*)mr_base_ + offset;
    }

    // Stack-style allocator: must be freed in LIFO order
    void* alloc_slice(size_t bytes, size_t alignment = 256) {
        // TODO
        return nullptr;
    }

    // Free a slice: must be in LIFO order
    void free_slice(void* p, size_t bytes) {
        // TODO
    }

    // Get available size of the memory pool
    size_t get_size_available() const {
        // TODO
        return 0;
    }

    // Get total size of the memory pool
    size_t get_size_total() const {
        // TODO
        return 0;
    }


    // Verify the memory pool 
    bool verify() const {
        // Write a pattern via GPU and reading back
        const uint8_t pattern = 0xAB;
        CC(cudaMemset(dev_base_, pattern, size_));
        CC(cudaDeviceSynchronize());

        auto* p = static_cast<const uint8_t*>(mr_base_);
        for (size_t i = 0; i < size_; i += 4096) {
            if (p[i] != pattern) return false;
        }

        CC(cudaMemset(dev_base_, 0, size_));
        CC(cudaDeviceSynchronize());
        for (size_t i = 0; i < size_; i += 4096) {
            if (p[i] != 0) return false;
        }
        return true;
    }

    // Accessors
    uint32_t lkey()      const { return mr_->lkey; }
    uint32_t rkey()      const { return mr_->rkey; }
    void*    base()      const { return mr_base_; }
    void*    dev_base()  const { return dev_base_; }
    size_t   size()      const { return size_; }
    bool     is_host()   const { return host_; }

private:
    ibv_context* ctx_     = nullptr;  // RDMA device handle
    ibv_pd*      pd_      = nullptr;  // protection domain
    ibv_mr*      mr_      = nullptr;  // single memory registration
    void*        mr_base_ = nullptr;  // MR-registered base ptr
    void*        dev_base_= nullptr;  // GPU-accessible ptr
    size_t       size_    = 0;
    bool         host_    = false;    // true = fallback host path
    char*        head_    = nullptr;  // stack head for slices
    char*        end_     = nullptr;  // end of pool
};

static void parse_optional_args(int argc, char** argv,
                                size_t& pool_size_mb) {
    // Accept:
    //   --pool-size-mb <megabytes>
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--pool-size-mb") == 0 && i + 1 < argc) {
            pool_size_mb = static_cast<size_t>(std::strtoull(argv[i + 1], nullptr, 10));
            i++;
        }
    }
}

static bool runTest(size_t pool_size_mb) {
    try {
        CC(cudaSetDeviceFlags(cudaDeviceMapHost));
        CC(cudaSetDevice(0));
        CC(cudaFree(0));

        MemoryPool pool(pool_size_mb * 1024ULL * 1024);
        if (!pool.verify()) {
            throw std::runtime_error("memory verification failed");
        }

        size_t total_before = pool.get_size_total();
        size_t avail_before = pool.get_size_available();
        if (avail_before != total_before) {
            throw std::runtime_error("unexpected initial available size");
        }

        constexpr size_t LEN1 = 4096;
        constexpr size_t LEN2 = 8192;
        void* s1 = pool.alloc_slice(LEN1);
        void* s2 = pool.alloc_slice(LEN2);

        size_t avail_mid = pool.get_size_available();
        if (avail_mid >= avail_before) {
            throw std::runtime_error("available size did not decrease after alloc");
        }

        void* dev_s1 = (char*)pool.dev_base() + ((char*)s1 - (char*)pool.base());
        void* dev_s2 = (char*)pool.dev_base() + ((char*)s2 - (char*)pool.base());

        // Use byte patterns to verify slice visibility and isolation
        CC(cudaMemset(dev_s1, 0x11, LEN1));
        CC(cudaMemset(dev_s2, 0x22, LEN2));
        CC(cudaDeviceSynchronize());

        // If host reads match the GPU-written patterns, the slice mapping is correct
        auto* p1 = static_cast<const uint8_t*>(s1);
        auto* p2 = static_cast<const uint8_t*>(s2);
        for (size_t i = 0; i < LEN1; i++) {
            if (p1[i] != 0x11) throw std::runtime_error("slice1 verify failed");
        }
        for (size_t i = 0; i < LEN2; i++) {
            if (p2[i] != 0x22) throw std::runtime_error("slice2 verify failed");
        }

        pool.free_slice(s2, LEN2);
        pool.free_slice(s1, LEN1);

        if (pool.get_size_available() != total_before) {
            throw std::runtime_error("available size did not recover after free");
        }

        // Verify allocating more than available memory throws an error
        try {
            size_t too_big = pool.get_size_available() + 1;
            (void)pool.alloc_slice(too_big);
            throw std::runtime_error("expected out-of-memory not thrown");
        } catch (const std::runtime_error&) {
            // Correctly threw out-of-memory error
        }

        constexpr size_t OFF = 64 * 1024, LEN = 4096;
        void* slice = pool.mr_slice(OFF, LEN);
        void* dev_slice = (char*)pool.dev_base() + OFF;

        CC(cudaMemset(dev_slice, 0x42, LEN));
        CC(cudaDeviceSynchronize());

        auto* p = static_cast<const uint8_t*>(slice);
        for (size_t i = 0; i < LEN; i++) {
            if (p[i] != 0x42) {
                throw std::runtime_error("mr_slice verify failed");
            }
        }
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Default memory pool size is 256 MB
    size_t pool_size_mb = 256;
    // parse pool size as an optional argument
    parse_optional_args(argc, argv, pool_size_mb);

    if (pool_size_mb < 128 || pool_size_mb > 16384) {
        fprintf(stderr, "pool size must be 128..16384 MB (got %zu)\n", pool_size_mb);
        return 1;
    }

    bool ok = runTest(pool_size_mb);
    if (ok) {
        printf("{\"Correctness\":\"PASS\"}\n");
        return 0;
    }
    printf("{\"Correctness\":\"FAIL\"}\n");
    return 1;
}
