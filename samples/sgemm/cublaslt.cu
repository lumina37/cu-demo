#include <cublasLt.h>
#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <vector>

#include "../cud_helper.hpp"

int main() {
    constexpr std::array SIZES{2048, 3072, 4096};
    constexpr int HEATUP_TIMES = 1;
    constexpr int PERF_TIMES = 3;

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);

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

        cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
        cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, M, K, M);
        cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, K);
        cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, M);

        constexpr int expectAlgoCount = 1;
        std::array<cublasLtMatmulHeuristicResult_t, expectAlgoCount> heuristicResult;
        int returnedResults = 0;

        cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, layoutA, layoutB, layoutC, layoutC, preference,
                                       expectAlgoCount,
                                       heuristicResult.data(), &returnedResults);

        if (returnedResults == 0) {
            std::cerr << "No suitable SIMT algorithm found!" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < HEATUP_TIMES; i++) {
            cublasLtMatmul(ltHandle, operationDesc, &alpha, deviceB, layoutB, deviceA, layoutA, &beta, deviceC, layoutC,
                           deviceC, layoutC,
                           &heuristicResult[0].algo, nullptr, 0, nullptr);
            cudaDeviceSynchronize();
        }

        std::vector<float> elapsedTimes;
        float elapsedTime = 0;
        for (int i = 0; i < PERF_TIMES; i++) {
            cudaEventRecord(evBegin);
            cublasLtMatmul(ltHandle, operationDesc, &alpha, deviceB, layoutB, deviceA, layoutA, &beta, deviceC, layoutC,
                           deviceC, layoutC,
                           &heuristicResult[0].algo, nullptr, 0, nullptr);
            cudaEventRecord(evEnd);
            cudaEventSynchronize(evEnd);
            cudaEventElapsedTime(&elapsedTime, evBegin, evEnd);
            elapsedTimes.push_back(elapsedTime);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)M * N * K * 2;
        const float meanTflops = macs / meanTime / 1e9;

        std::cout << "=============================" << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "Performance: " << meanTflops << " tflops" << std::endl;

        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
    }

    cudaEventDestroy(evBegin);
    cudaEventDestroy(evEnd);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
}