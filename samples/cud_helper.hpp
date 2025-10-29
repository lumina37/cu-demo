#pragma once

#include <chrono>
#include <utility>

namespace fs = std::filesystem;

#define CHECK_CUDA(call)                                                                                     \
    do {                                                                                                     \
        cudaError_t err = call;                                                                              \
        if (err != cudaSuccess) {                                                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) \
                      << std::endl;                                                                          \
            exit(EXIT_FAILURE);                                                                              \
        }                                                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                                                  \
    do {                                                                                                    \
        cublasStatus_t status = call;                                                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                              \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    } while (0)

std::pair<float, float> meanStd(const std::vector<float>& data) {
    float mean = 0.0;
    float acc2 = 0.0;

    for (size_t i = 0; i < data.size(); ++i) {
        float delta = data[i] - mean;
        mean += delta / float(i + 1);
        float delta2 = data[i] - mean;
        acc2 += delta * delta2;
    }

    float variance = acc2 / (float)data.size();
    return {mean, std::sqrt(variance)};
}
