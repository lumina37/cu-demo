#pragma once

#include <cassert>
#include <cstdio>
#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../../cud_helper.hpp"

using uint = unsigned int;

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

    auto loadGlobalToShared = [&](uint globalCoordX, uint globalCoordY, uint globalExtentX, uint globalExtentY,
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

    auto computeSubThreadTile = [&](uint iterTM, uint iterVecTN, uint stage) {
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

                            regAccumulator[regCoordM][regCoordVecN].x += regA[iterVecSubTK].x * regB[
                                regBaseKB + 0][iterVecSubTN].x;
                            regAccumulator[regCoordM][regCoordVecN].y += regA[iterVecSubTK].x * regB[
                                regBaseKB + 0][iterVecSubTN].y;
                            regAccumulator[regCoordM][regCoordVecN].z += regA[iterVecSubTK].x * regB[
                                regBaseKB + 0][iterVecSubTN].z;
                            regAccumulator[regCoordM][regCoordVecN].w += regA[iterVecSubTK].x * regB[
                                regBaseKB + 0][iterVecSubTN].w;

                            regAccumulator[regCoordM][regCoordVecN].x += regA[iterVecSubTK].y * regB[
                                regBaseKB + 1][iterVecSubTN].x;
                            regAccumulator[regCoordM][regCoordVecN].y += regA[iterVecSubTK].y * regB[
                                regBaseKB + 1][iterVecSubTN].y;
                            regAccumulator[regCoordM][regCoordVecN].z += regA[iterVecSubTK].y * regB[
                                regBaseKB + 1][iterVecSubTN].z;
                            regAccumulator[regCoordM][regCoordVecN].w += regA[iterVecSubTK].y * regB[
                                regBaseKB + 1][iterVecSubTN].w;

                            regAccumulator[regCoordM][regCoordVecN].x += regA[iterVecSubTK].z * regB[
                                regBaseKB + 2][iterVecSubTN].x;
                            regAccumulator[regCoordM][regCoordVecN].y += regA[iterVecSubTK].z * regB[
                                regBaseKB + 2][iterVecSubTN].y;
                            regAccumulator[regCoordM][regCoordVecN].z += regA[iterVecSubTK].z * regB[
                                regBaseKB + 2][iterVecSubTN].z;
                            regAccumulator[regCoordM][regCoordVecN].w += regA[iterVecSubTK].z * regB[
                                regBaseKB + 2][iterVecSubTN].w;

                            regAccumulator[regCoordM][regCoordVecN].x += regA[iterVecSubTK].w * regB[
                                regBaseKB + 3][iterVecSubTN].x;
                            regAccumulator[regCoordM][regCoordVecN].y += regA[iterVecSubTK].w * regB[
                                regBaseKB + 3][iterVecSubTN].y;
                            regAccumulator[regCoordM][regCoordVecN].z += regA[iterVecSubTK].w * regB[
                                regBaseKB + 3][iterVecSubTN].z;
                            regAccumulator[regCoordM][regCoordVecN].w += regA[iterVecSubTK].w * regB[
                                regBaseKB + 3][iterVecSubTN].w;
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


void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha,
               float* A, float* B, float beta, float* C) {
    // cuBLAS uses column-major order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                 N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


void runMySgemm(int M, int N, int K, float* A, float* B, float* C) {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BK = 16;
    constexpr uint32_t TM = 16;
    constexpr uint32_t TN = 8;
    constexpr uint32_t TK = 16;
    constexpr uint32_t STM = 16;
    constexpr uint32_t STN = 8;
    constexpr uint32_t STK = 8;
    dim3 gridDim(N / BN, M / BM);
    dim3 blockDim(BN / TN, BM / TM);
    sgemmSubTile<BM, BN, BK, TM, TN, TK, STM, STN, STK> <<<gridDim, blockDim>>>(
        M, N, K, (float4*)A, (float4*)B, (float4*)C);
}

void randomizeMat(float* mat, int N) {
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        mat[i] = dist(rdEngine);
    }
}

bool verifyMat(float* matRef, float* matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (isnan(diff) || diff > 0.01) {
            printf("expect %5.2f, get %5.2f at %d\n", matRef[i], matOut[i], i);
            return false;
        }
    }
    return true;
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t evBegin, evEnd;
    cudaEventCreate(&evBegin);
    cudaEventCreate(&evEnd);

    constexpr std::array SIZES{2048, 3072, 4096};

    int M, N, K;
    const int maxSize = SIZES.back();

    const float alpha = 1.0, beta = 0.0;

    float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr, *deviceCRef = nullptr;

    std::vector<float> hostA(maxSize * maxSize);
    std::vector<float> hostB(maxSize * maxSize);
    std::vector<float> hostC(maxSize * maxSize);
    std::vector<float> hostCRef(maxSize * maxSize);

    randomizeMat(hostA.data(), maxSize * maxSize);
    randomizeMat(hostB.data(), maxSize * maxSize);

    cudaMalloc(&deviceA, sizeof(float) * maxSize * maxSize);
    cudaMalloc(&deviceB, sizeof(float) * maxSize * maxSize);
    cudaMalloc(&deviceC, sizeof(float) * maxSize * maxSize);
    cudaMalloc(&deviceCRef, sizeof(float) * maxSize * maxSize);

    cudaMemcpy(deviceA, hostA.data(), sizeof(float) * maxSize * maxSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB.data(), sizeof(float) * maxSize * maxSize, cudaMemcpyHostToDevice);

    const int PERF_TIMES = 3;
    for (const int size : SIZES) {
        M = N = K = size;
        runCublas(handle, M, N, K, alpha, deviceA, deviceB, beta, deviceCRef);
        runMySgemm(M, N, K, deviceA, deviceB, deviceC);
        cudaDeviceSynchronize();

        cudaMemcpy(hostC.data(), deviceC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(hostCRef.data(), deviceCRef, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        if (!verifyMat(hostCRef.data(), hostC.data(), M * N)) {
            std::cout << "verification failed" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<float> elapsedTimes;
        for (int i = 0; i < PERF_TIMES; i++) {
            cudaEventRecord(evBegin);
            runMySgemm(M, N, K, deviceA, deviceB, deviceC);
            cudaEventRecord(evEnd);
            cudaEventSynchronize(evEnd);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, evBegin, evEnd);
            elapsedTimes.push_back(elapsedTime);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)M * N * K * 2;
        const float meanTflops = macs / meanTime / 1e9;

        std::cout << "=============================" << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "Performance: " << meanTflops << " tflops" << std::endl;
    }

    // Free up CPU and GPU space
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFree(deviceCRef);
    cublasDestroy(handle);
}