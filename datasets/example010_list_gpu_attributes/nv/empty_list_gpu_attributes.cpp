/*
 * empty_list_gpu_attributes.cpp
 *
 * Purpose:
 *   This file is intentionally incomplete. Fill in GPU attribute collection logic.
 *
 * Notes:
 *   - Correctness-only example (no performance benchmark required).
 *   - Prints exactly one JSON object to stdout.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

class GPUInfoCollector {
public:
    struct GPUAttributes {
        std::string name;
        size_t totalGlobalMemBytes = 0;
        int smCount = 0;
        int major = 0;
        int minor = 0;
        int warpSize = 0;
        int maxThreadsPerBlock = 0;
        int clockRateKHz = 0;          // CUDA reports kHz
        int memoryClockRateKHz = 0;    // CUDA reports kHz
        int memoryBusWidthBits = 0;
        size_t sharedMemPerBlockBytes = 0;
    };

    bool collect() {
        gpus_.clear();

        // TODO:
        // 1) Call cudaGetDeviceCount(&deviceCount)
        // 2) If error or deviceCount <= 0, return false
        // 3) For each device:
        //    - Call cudaGetDeviceProperties(&prop, dev)
        //    - Populate a GPUAttributes struct from prop
        //    - Push it into gpus_
        // 4) Return validateCollectedAttributes()

        return false;
    }

    const std::vector<GPUAttributes>& gpus() const { return gpus_; }

private:
    bool validateCollectedAttributes() const {
        // TODO:
        // Validate that gpus_ is non-empty and each GPUAttributes has meaningful values.
        // Suggested checks:
        // - name not empty
        // - totalGlobalMemBytes > 0
        // - smCount > 0
        // - warpSize > 0
        // - maxThreadsPerBlock > 0
        // - major/minor non-negative
        // - at least one of (clockRateKHz, memoryClockRateKHz, memoryBusWidthBits) > 0

        return false;
    }

    std::vector<GPUAttributes> gpus_;
};

static void printPassWithAttributes(const GPUInfoCollector& collector) {
    std::cout << "{ \"Correctness\": \"PASS\", \"gpu_count\": " << collector.gpus().size() << ", \"gpus\": [";

    const auto& gpus = collector.gpus();
    for (size_t i = 0; i < gpus.size(); ++i) {
        const auto& g = gpus[i];
        if (i) std::cout << ", ";

        // NOTE: prop.name should not contain quotes normally; keep it simple here.
        std::cout
            << "{"
            << "\"name\":\"" << g.name << "\","
            << "\"totalGlobalMemBytes\":" << g.totalGlobalMemBytes << ","
            << "\"smCount\":" << g.smCount << ","
            << "\"major\":" << g.major << ","
            << "\"minor\":" << g.minor << ","
            << "\"warpSize\":" << g.warpSize << ","
            << "\"maxThreadsPerBlock\":" << g.maxThreadsPerBlock << ","
            << "\"clockRateKHz\":" << g.clockRateKHz << ","
            << "\"memoryClockRateKHz\":" << g.memoryClockRateKHz << ","
            << "\"memoryBusWidthBits\":" << g.memoryBusWidthBits << ","
            << "\"sharedMemPerBlockBytes\":" << g.sharedMemPerBlockBytes
            << "}";
    }

    std::cout << "] }";
}


static void printFail() {
    std::cout << "{ \"Correctness\": \"FAIL\" }";
}

static void runTest() {
    GPUInfoCollector collector;
    const bool ok = collector.collect();

    // NOTE: Do NOT print device info; stdout must contain exactly one JSON object.
    if (ok) {
        printPassWithAttributes(collector);
    } else {
        printFail();
    }
}

int main() {
    runTest();
    return 0;
}
