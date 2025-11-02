#include <array>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include "../cud_helper.hpp"

using TTensor = cutlass::HostTensor<float, cutlass::layout::RowMajor>;

void cublasSgemmHelper(cublasHandle_t handle, const cutlass::gemm::GemmCoord& problemSize, TTensor& tensorA,
                       TTensor& tensorB,
                       TTensor& tensorC) {
    constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, problemSize.n(), problemSize.m(), problemSize.k(), &alpha,
                 tensorB.device_data(),
                 CUDA_R_32F,
                 problemSize.n(), tensorA.device_data(), CUDA_R_32F, problemSize.k(), &beta, tensorC.device_data(),
                 CUDA_R_32F, problemSize.n(),
                 CUBLAS_COMPUTE_32F,
                 algo);
}

int main() {
    constexpr std::array SIZES{1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240};
    constexpr int HEATUP_TIMES = 1;
    constexpr int PERF_TIMES = 3;

    cublasHandle_t handle;
    cublasCreate(&handle);

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

        for (int i = 0; i < HEATUP_TIMES; i++) {
            cublasSgemmHelper(handle, problemSize, tensorA, tensorB, tensorC);
            cudaDeviceSynchronize();
        }

        std::vector<float> elapsedTimes;
        for (int i = 0; i < PERF_TIMES; i++) {
            cudaEventRecord(evBegin);

            cublasSgemmHelper(handle, problemSize, tensorA, tensorB, tensorC);

            cudaEventRecord(evEnd);
            cudaEventSynchronize(evEnd);

            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, evBegin, evEnd);

            elapsedTimes.push_back(elapsedTime);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)size * size * size * 2;
        const float meanTflops = macs / meanTime / 1e9;

        std::cout << "=============================" << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "Performance: " << meanTflops << " tflops" << std::endl;
    }

    cudaEventDestroy(evBegin);
    cudaEventDestroy(evEnd);
    cublasDestroy(handle);
}