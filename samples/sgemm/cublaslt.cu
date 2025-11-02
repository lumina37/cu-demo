#include <array>
#include <iostream>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include "../cud_helper.hpp"

using TTensor = cutlass::HostTensor<float, cutlass::layout::RowMajor>;

int main() {
    constexpr std::array SIZES{1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240};
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

    constexpr int MAX_SIZE = SIZES.back();
    TTensor tensorA({MAX_SIZE, MAX_SIZE});
    TTensor tensorB({MAX_SIZE, MAX_SIZE});
    TTensor tensorC({MAX_SIZE, MAX_SIZE});

    cutlass::reference::host::TensorFillRandomUniform(tensorA.host_view(), 42, 1.f, 0.f);
    cutlass::reference::host::TensorFillRandomUniform(tensorB.host_view(), 42, 1.f, 0.f);

    tensorA.sync_device();
    tensorB.sync_device();

    for (const int size : SIZES) {
        cutlass::gemm::GemmCoord problemSize(size, size, size);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
        cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, problemSize.m(), problemSize.k(), problemSize.m());
        cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, problemSize.k(), problemSize.n(), problemSize.k());
        cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, problemSize.m(), problemSize.n(), problemSize.m());

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
            cublasLtMatmul(ltHandle, operationDesc, &alpha, tensorB.device_data(), layoutB, tensorA.device_data(),
                           layoutA, &beta,
                           tensorC.device_data(), layoutC,
                           tensorC.device_data(), layoutC,
                           &heuristicResult[0].algo, nullptr, 0, nullptr);
            cudaDeviceSynchronize();
        }

        std::vector<float> elapsedTimes;
        float elapsedTime = 0;
        for (int i = 0; i < PERF_TIMES; i++) {
            cudaEventRecord(evBegin);
            cublasLtMatmul(ltHandle, operationDesc, &alpha, tensorB.device_data(), layoutB, tensorA.device_data(),
                           layoutA, &beta,
                           tensorC.device_data(), layoutC,
                           tensorC.device_data(), layoutC,
                           &heuristicResult[0].algo, nullptr, 0, nullptr);
            cudaEventRecord(evEnd);
            cudaEventSynchronize(evEnd);
            cudaEventElapsedTime(&elapsedTime, evBegin, evEnd);
            elapsedTimes.push_back(elapsedTime);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)size * size * size * 2;
        const float meanTflops = macs / meanTime / 1e9;

        std::cout << "=============================" << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "Performance: " << meanTflops << " tflops" << std::endl;

        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
    }

    cudaEventDestroy(evBegin);
    cudaEventDestroy(evEnd);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
}