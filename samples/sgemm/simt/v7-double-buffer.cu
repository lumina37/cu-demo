#include <cassert>
#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/gemm.h>

#include "../../cud_helper.hpp"

using uint = unsigned int;

__device__ inline void fma(float4 a, float b, float4& c) {
    c = make_float4(__fmaf_rn(a.x, b, c.x),
                    __fmaf_rn(a.y, b, c.y),
                    __fmaf_rn(a.z, b, c.z),
                    __fmaf_rn(a.w, b, c.w));
}

template <
    uint BLOCK_TILE_M = 128,
    uint BLOCK_TILE_N = 128,
    uint BLOCK_TILE_K = 128,
    uint THREAD_TILE_M = 16,
    uint THREAD_TILE_N = 16,
    uint THREAD_TILE_K = 16,
    uint THREAD_SUBTILE_M = 8,
    uint THREAD_SUBTILE_N = 8,
    uint THREAD_SUBTILE_K = 8
>
__global__ void sgemmSubTile(uint M, uint N, uint K,
                             const float4* __restrict__ srcMatA,
                             const float4* __restrict__ srcMatB,
                             float4* __restrict__ dstMat) {
    constexpr uint BLOCK_TILE_VEC_N = BLOCK_TILE_N / 4;
    constexpr uint BLOCK_TILE_VEC_K = BLOCK_TILE_K / 4;
    constexpr uint THREAD_TILE_VEC_N = THREAD_TILE_N / 4;
    constexpr uint THREAD_SUBTILE_VEC_N = THREAD_SUBTILE_N / 4;
    constexpr uint THREAD_SUBTILE_VEC_K = THREAD_SUBTILE_K / 4;
    constexpr uint STAGES = 2;

    // Shared memory
    __shared__ float4 sharedA[STAGES][BLOCK_TILE_M][BLOCK_TILE_VEC_K];
    __shared__ float4 sharedB[STAGES][BLOCK_TILE_K][BLOCK_TILE_VEC_N];

    // Thread indices
    const uint localIndex = threadIdx.y * blockDim.x + threadIdx.x;
    const uint groupThreadCount = blockDim.x * blockDim.y;

    // Accumulator registers
    float4 regAccumulator[THREAD_TILE_M][THREAD_TILE_VEC_N];

    // Zero-fill accumulator
#pragma unroll
    for (uint iterTM = 0; iterTM < THREAD_TILE_M; iterTM++) {
#pragma unroll
        for (uint iterVecTN = 0; iterVecTN < THREAD_TILE_VEC_N; iterVecTN++) {
            regAccumulator[iterTM][iterVecTN] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const auto loadGlobalToShared = [&](uint globalCoordX, uint globalCoordY, uint globalExtentX, uint globalExtentY,
                                        uint globalRowStride, bool loadA, uint stage) {
        const uint loadsPerThread = (globalExtentX * globalExtentY) / groupThreadCount;

        for (uint i = 0; i < loadsPerThread; i++) {
            const uint linearIdx = i * groupThreadCount + localIndex;
            const uint srcOffsetX = linearIdx % globalExtentX;
            const uint srcOffsetY = linearIdx / globalExtentX;
            const uint srcCoordX = globalCoordX + srcOffsetX;
            const uint srcCoordY = globalCoordY + srcOffsetY;
            const uint srcIndex = srcCoordY * globalRowStride + srcCoordX;

            if (loadA) {
                sharedA[stage][srcOffsetY][srcOffsetX] = srcMatA[srcIndex];
            } else {
                sharedB[stage][srcOffsetY][srcOffsetX] = srcMatB[srcIndex];
            }
        }
    };

    // Main loop over K dimension
    const uint blockBaseM = blockIdx.y * BLOCK_TILE_M;
    const uint blockBaseVecN = blockIdx.x * BLOCK_TILE_VEC_N;

    const auto computeSubThreadTile = [&](uint iterTM, uint iterVecTN, uint stage) {
        float4 regA[THREAD_SUBTILE_VEC_K];
        float4 regB[THREAD_SUBTILE_K][THREAD_SUBTILE_VEC_N];

#pragma unroll
        for (uint iterBK = 0; iterBK < BLOCK_TILE_K; iterBK += THREAD_TILE_K) {
#pragma unroll
            for (uint iterTK = 0; iterTK < THREAD_TILE_K; iterTK += THREAD_SUBTILE_K) {
#pragma unroll
                for (uint iterSubTK = 0; iterSubTK < THREAD_SUBTILE_K; iterSubTK++) {
                    const uint sharedCoordKB = iterBK + iterTK + iterSubTK;
#pragma unroll
                    for (uint iterVecSubTN = 0; iterVecSubTN < THREAD_SUBTILE_VEC_N; iterVecSubTN++) {
                        const uint sharedCoordVecNB = (iterVecTN + iterVecSubTN) * blockDim.x + threadIdx.x;
                        regB[iterSubTK][iterVecSubTN] = sharedB[stage][sharedCoordKB][sharedCoordVecNB];
                    }
                }

#pragma unroll
                for (uint iterSubTM = 0; iterSubTM < THREAD_SUBTILE_M; iterSubTM++) {
                    const uint regCoordM = iterTM + iterSubTM;
                    const uint sharedCoordMA = (iterTM + iterSubTM) * blockDim.y + threadIdx.y;
#pragma unroll
                    for (uint iterVecSubTK = 0; iterVecSubTK < THREAD_SUBTILE_VEC_K; iterVecSubTK++) {
                        const uint sharedCoordVecKA = (iterBK + iterTK) / 4 + iterVecSubTK;
                        regA[iterVecSubTK] = sharedA[stage][sharedCoordMA][sharedCoordVecKA];
                    }

#pragma unroll
                    for (uint iterVecSubTN = 0; iterVecSubTN < THREAD_SUBTILE_VEC_N; iterVecSubTN++) {
                        const uint regCoordVecN = iterVecTN + iterVecSubTN;
#pragma unroll
                        for (uint iterVecSubTK = 0; iterVecSubTK < THREAD_SUBTILE_VEC_K; iterVecSubTK++) {
                            const uint regBaseKB = iterVecSubTK * 4;

                            fma(regB[regBaseKB + 0][iterVecSubTN], regA[iterVecSubTK].x,
                                regAccumulator[regCoordM][regCoordVecN]);
                            fma(regB[regBaseKB + 1][iterVecSubTN], regA[iterVecSubTK].y,
                                regAccumulator[regCoordM][regCoordVecN]);
                            fma(regB[regBaseKB + 2][iterVecSubTN], regA[iterVecSubTK].z,
                                regAccumulator[regCoordM][regCoordVecN]);
                            fma(regB[regBaseKB + 3][iterVecSubTN], regA[iterVecSubTK].w,
                                regAccumulator[regCoordM][regCoordVecN]);
                        }
                    }
                }
            }
        }
    };

    auto computeWithShared = [&](uint stage) {
#pragma unroll
        for (uint iterTM = 0; iterTM < THREAD_TILE_M; iterTM += THREAD_SUBTILE_M) {
#pragma unroll
            for (uint iterVecTN = 0; iterVecTN < THREAD_TILE_VEC_N; iterVecTN += THREAD_SUBTILE_VEC_N) {
                computeSubThreadTile(iterTM, iterVecTN, stage);
            }
        }
    };

    uint stage = 0;
    loadGlobalToShared(0, blockBaseM, BLOCK_TILE_VEC_K, BLOCK_TILE_M, K / 4, true, 0);
    loadGlobalToShared(blockBaseVecN, 0, BLOCK_TILE_VEC_N, BLOCK_TILE_K, N / 4, false, 0);
    __syncthreads();

#pragma unroll
    for (uint iterK = BLOCK_TILE_K; iterK < K; iterK += BLOCK_TILE_K) {
        const uint nextStage = (stage + 1) % STAGES;

        loadGlobalToShared(iterK / 4, blockBaseM, BLOCK_TILE_VEC_K, BLOCK_TILE_M, K / 4, true, nextStage);
        loadGlobalToShared(blockBaseVecN, iterK, BLOCK_TILE_VEC_N, BLOCK_TILE_K, N / 4, false, nextStage);

        computeWithShared(stage);

        stage = nextStage;
        __syncthreads();
    }

    computeWithShared(stage);

    // Store results to global memory
    const uint globalBaseM = blockIdx.y * BLOCK_TILE_M + threadIdx.y;
    const uint globalBaseVecN = blockIdx.x * BLOCK_TILE_VEC_N + threadIdx.x;

#pragma unroll
    for (uint iterTM = 0; iterTM < THREAD_TILE_M; iterTM++) {
#pragma unroll
        for (uint iterVecTN = 0; iterVecTN < THREAD_TILE_VEC_N; iterVecTN++) {
            const uint globalCoordM = globalBaseM + iterTM * blockDim.y;
            const uint globalCoordVecN = globalBaseVecN + iterVecTN * blockDim.x;
            const uint dstIdx = globalCoordM * (N / 4) + globalCoordVecN;
            dstMat[dstIdx] = regAccumulator[iterTM][iterVecTN];
        }
    }
}

using TTensor = cutlass::HostTensor<float, cutlass::layout::RowMajor>;
using TView = cutlass::TensorView<float, cutlass::layout::RowMajor>;

void sgemmHelper(const cutlass::gemm::GemmCoord& problemSize, TTensor& tensorA,
                 TTensor& tensorB,
                 TTensor& tensorC) {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BK = 16;
    constexpr uint32_t TM = 16;
    constexpr uint32_t TN = 8;
    constexpr uint32_t TK = 16;
    constexpr uint32_t STM = 16;
    constexpr uint32_t STN = 8;
    constexpr uint32_t STK = 8;
    dim3 gridDim(problemSize.n() / BN, problemSize.m() / BM);
    dim3 blockDim(BN / TN, BM / TM);
    sgemmSubTile<BM, BN, BK, TM, TN, TK, STM, STN, STK> <<<gridDim, blockDim>>>(
        problemSize.m(), problemSize.n(), problemSize.k(), reinterpret_cast<float4*>(tensorA.device_data()),
        reinterpret_cast<float4*>(tensorB.device_data()),
        reinterpret_cast<float4*>(tensorC.device_data()));
}

bool verifyMat(float* matRef, float* matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (isnan(diff) || diff > 0.01) {
            std::cout << "expect " << matRef[i] << ", got " << matOut[i] << " at " << i << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    cutlass::reference::device::Gemm<float,
                                     cutlass::layout::RowMajor,
                                     float,
                                     cutlass::layout::RowMajor,
                                     float,
                                     cutlass::layout::RowMajor,
                                     float,
                                     float> gemmRefOp;

    cudaEvent_t evBegin, evEnd;
    cudaEventCreate(&evBegin);
    cudaEventCreate(&evEnd);

    constexpr std::array SIZES{1024, 2048, 3072, 4096};

    const int PERF_TIMES = 3;
    for (const int size : SIZES) {
        cutlass::gemm::GemmCoord problemSize(size, size, size);

        TTensor tensorA(problemSize.mk());
        TTensor tensorB(problemSize.kn());
        TTensor tensorC(problemSize.mn());
        TTensor tensorCRef(problemSize.mn());

        cutlass::reference::host::TensorFillRandomUniform(tensorA.host_view(), 42, 1.f, 0.f);
        cutlass::reference::host::TensorFillRandomUniform(tensorB.host_view(), 42, 1.f, 0.f);

        tensorA.sync_device();
        tensorB.sync_device();

        gemmRefOp(problemSize,
                  1.f,
                  tensorA.device_ref(),
                  tensorB.device_ref(),
                  0.f,
                  tensorCRef.device_ref(),
                  tensorCRef.device_ref());
        cudaDeviceSynchronize();

        sgemmHelper(problemSize, tensorA, tensorB, tensorC);
        cudaDeviceSynchronize();

        tensorC.sync_host();
        tensorCRef.sync_host();

        if (!verifyMat(tensorCRef.host_data(), tensorC.host_data(), problemSize.m() * problemSize.n())) {
            std::cout << "verification failed" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<float> elapsedTimes;
        for (int i = 0; i < PERF_TIMES; i++) {
            cudaEventRecord(evBegin);

            sgemmHelper(problemSize, tensorA, tensorB, tensorC);

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
}