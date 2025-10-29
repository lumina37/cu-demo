#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <thread>
#include <vector>

#include "../cud_helper.hpp"

int main() {
    constexpr std::array SIZES{1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240};
    constexpr int HEATUP_TIMES = 1;
    constexpr int PERF_TIMES = 3;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t evBegin, evEnd;
    cudaEventCreate(&evBegin);
    cudaEventCreate(&evEnd);

    for (const int size : SIZES) {
        const int M = size;
        const int N = size;
        const int K = size;

        float *deviceA, *deviceB, *deviceC;
        cudaMalloc(&deviceA, M * K * sizeof(float));
        cudaMalloc(&deviceB, K * N * sizeof(float));
        cudaMalloc(&deviceC, M * N * sizeof(float));

        std::vector hostA(M * K, 1.0f);
        std::vector hostB(K * N, 2.0f);

        cudaMemcpy(deviceA, hostA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

        for (int i = 0; i < HEATUP_TIMES; i++) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, deviceB, CUDA_R_32F,
                         N, deviceA, CUDA_R_32F, K, &beta, deviceC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         algo);
            cudaDeviceSynchronize();
        }

        std::vector<float> elapsedTimes;
        for (int i = 0; i < PERF_TIMES; i++) {
            cudaEventRecord(evBegin);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, deviceB, CUDA_R_32F,
                         N, deviceA, CUDA_R_32F, K, &beta, deviceC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         algo);
            cudaEventRecord(evEnd);
            cudaEventSynchronize(evEnd);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, evBegin, evEnd);
            elapsedTimes.push_back(elapsedTime);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)M * N * K * 2;
        const float meanTflops = macs / meanTime / 1e9;

        // std::cout << "=============================" << std::endl;
        // std::cout << "Size: " << size << std::endl;
        // std::cout << "Performance: " << meanTflops << " tflops" << std::endl;
        std::cout << meanTflops << std::endl;

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
    }

    cudaEventDestroy(evBegin);
    cudaEventDestroy(evEnd);
    cublasDestroy(handle);
}