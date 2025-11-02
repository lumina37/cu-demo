#include <array>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"

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
        cutlass::gemm::GemmCoord gemmCoord(size, size, size);

        cutlass::HostTensor<float, cutlass::layout::RowMajor> deviceA(gemmCoord.mk());
        cutlass::HostTensor<float, cutlass::layout::RowMajor> deviceB(gemmCoord.kn());
        cutlass::HostTensor<float, cutlass::layout::RowMajor> deviceC(gemmCoord.mn());

        cutlass::reference::host::TensorFillRandomUniform(deviceA.host_view(), 42, 1.f, 0.f, 0);
        cutlass::reference::host::TensorFillRandomUniform(deviceB.host_view(), 42, 1.f, 0.f, 0);

        deviceA.sync_device();
        deviceB.sync_device();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

        for (int i = 0; i < HEATUP_TIMES; i++) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, gemmCoord.n(), gemmCoord.m(), gemmCoord.k(), &alpha,
                         deviceB.device_data(),
                         CUDA_R_32F,
                         gemmCoord.n(), deviceA.device_data(), CUDA_R_32F, gemmCoord.k(), &beta, deviceC.device_data(),
                         CUDA_R_32F, gemmCoord.n(),
                         CUBLAS_COMPUTE_32F,
                         algo);
            cudaDeviceSynchronize();
        }

        std::vector<float> elapsedTimes;
        for (int i = 0; i < PERF_TIMES; i++) {
            cudaEventRecord(evBegin);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, gemmCoord.n(), gemmCoord.m(), gemmCoord.k(), &alpha,
                         deviceB.device_data(),
                         CUDA_R_32F,
                         gemmCoord.n(), deviceA.device_data(), CUDA_R_32F, gemmCoord.k(), &beta, deviceC.device_data(),
                         CUDA_R_32F, gemmCoord.n(),
                         CUBLAS_COMPUTE_32F,
                         algo);
            cudaEventRecord(evEnd);
            cudaEventSynchronize(evEnd);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, evBegin, evEnd);
            elapsedTimes.push_back(elapsedTime);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)size * size * size * 2;
        const float meanTflops = macs / meanTime / 1e9;

        // std::cout << "=============================" << std::endl;
        // std::cout << "Size: " << size << std::endl;
        // std::cout << "Performance: " << meanTflops << " tflops" << std::endl;
        std::cout << meanTflops << std::endl;
    }

    cudaEventDestroy(evBegin);
    cudaEventDestroy(evEnd);
    cublasDestroy(handle);
}