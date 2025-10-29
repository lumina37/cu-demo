#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>
#include <array>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <random>
#include <cuda_runtime.h>

#include "../../cud_helper.hpp"

template <
    uint32_t BLOCK_TILE_M = 128,
    uint32_t BLOCK_TILE_N = 128,
    uint32_t BLOCK_TILE_K = 128,
    uint32_t THREAD_TILE_M = 16,
    uint32_t THREAD_TILE_N = 16,
    uint32_t THREAD_TILE_K = 16,
    uint32_t THREAD_SUBTILE_M = 8,
    uint32_t THREAD_SUBTILE_N = 8,
    uint32_t THREAD_SUBTILE_K = 8
>
__global__ void sgemmSubTile(int M, int N, int K,
                             const float4* __restrict__ srcMatA,
                             const float4* __restrict__ srcMatB,
                             float4* __restrict__ dstMat) {
    constexpr int BLOCK_TILE_VEC_N = BLOCK_TILE_N / 4;
    constexpr int BLOCK_TILE_VEC_K = BLOCK_TILE_K / 4;
    constexpr int THREAD_TILE_VEC_N = THREAD_TILE_N / 4;
    constexpr int THREAD_SUBTILE_VEC_N = THREAD_SUBTILE_N / 4;
    constexpr int THREAD_SUBTILE_VEC_K = THREAD_SUBTILE_K / 4;

    // Shared memory
    __shared__ float4 sharedA[BLOCK_TILE_M][BLOCK_TILE_VEC_K];
    __shared__ float4 sharedB[BLOCK_TILE_K][BLOCK_TILE_VEC_N];

    // Thread indices
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int localIndex = tidy * blockDim.x + tidx;
    const int groupThreadCount = blockDim.x * blockDim.y;

    // Accumulator registers
    float4 regAccumulator[THREAD_TILE_M][THREAD_TILE_VEC_N];

    // Zero-fill accumulator
#pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_TILE_VEC_N; tn++) {
            regAccumulator[tm][tn] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Helper lambda for loading global to shared memory
    auto loadGlobalToShared = [&](int globalCoordX, int globalCoordY,
                                  int globalExtentX, int globalExtentY,
                                  int globalRowStride, bool loadA) {
        const int loadsPerThread = (globalExtentX * globalExtentY) / groupThreadCount;

        for (int i = 0; i < loadsPerThread; i++) {
            const int linearIdx = i * groupThreadCount + localIndex;
            const int srcOffsetX = linearIdx % globalExtentX;
            const int srcOffsetY = linearIdx / globalExtentX;
            const int srcCoordX = globalCoordX + srcOffsetX;
            const int srcCoordY = globalCoordY + srcOffsetY;
            const int srcIndex = srcCoordY * globalRowStride + srcCoordX;

            if (loadA) {
                sharedA[srcOffsetY][srcOffsetX] = srcMatA[srcIndex];
            } else {
                sharedB[srcOffsetY][srcOffsetX] = srcMatB[srcIndex];
            }
        }
    };

    // Main loop over K dimension
    const int blockBaseM = bidy * BLOCK_TILE_M;
    const int blockBaseVecN = bidx * BLOCK_TILE_VEC_N;
    const int blockSplitKCount = K / BLOCK_TILE_K;

    for (int iBlockK = 0; iBlockK < blockSplitKCount; iBlockK++) {
        const int blockBaseK = iBlockK * BLOCK_TILE_K;

        // Load tiles from global to shared memory
        loadGlobalToShared(blockBaseK / 4, blockBaseM, BLOCK_TILE_VEC_K, BLOCK_TILE_M, K / 4, true);
        loadGlobalToShared(blockBaseVecN, blockBaseK, BLOCK_TILE_VEC_N, BLOCK_TILE_K, N / 4, false);
        __syncthreads();

        // Compute with shared memory using sub-tiles
#pragma unroll
        for (int iterTM = 0; iterTM < THREAD_TILE_M; iterTM += THREAD_SUBTILE_M) {
#pragma unroll
            for (int iterVecTN = 0; iterVecTN < THREAD_TILE_VEC_N; iterVecTN += THREAD_SUBTILE_VEC_N) {
                // Sub-tile computation
                float4 regA[THREAD_SUBTILE_VEC_K];
                float4 regB[THREAD_SUBTILE_K][THREAD_SUBTILE_VEC_N];

#pragma unroll
                for (int iterBK = 0; iterBK < BLOCK_TILE_K; iterBK += THREAD_TILE_K) {
#pragma unroll
                    for (int iterTK = 0; iterTK < THREAD_TILE_K; iterTK += THREAD_SUBTILE_K) {
                        // Load B registers
#pragma unroll
                        for (int iterSubTK = 0; iterSubTK < THREAD_SUBTILE_K; iterSubTK++) {
                            const int sharedCoordYB = iterBK + iterTK + iterSubTK;
#pragma unroll
                            for (int iterVecSubTN = 0; iterVecSubTN < THREAD_SUBTILE_VEC_N; iterVecSubTN++) {
                                const int sharedCoordVecXB = (iterVecTN + iterVecSubTN) * blockDim.x + tidx;
                                regB[iterSubTK][iterVecSubTN] = sharedB[sharedCoordYB][sharedCoordVecXB];
                            }
                        }

                        // Compute sub-tile
#pragma unroll
                        for (int iterSubTM = 0; iterSubTM < THREAD_SUBTILE_M; iterSubTM++) {
                            const int regCoordY = iterTM + iterSubTM;
                            const int sharedCoordYA = (iterTM + iterSubTM) * blockDim.y + tidy;

                            // Load A registers
#pragma unroll
                            for (int iterVecSubTK = 0; iterVecSubTK < THREAD_SUBTILE_VEC_K; iterVecSubTK++) {
                                const int sharedCoordVecXA = (iterBK + iterTK) / 4 + iterVecSubTK;
                                regA[iterVecSubTK] = sharedA[sharedCoordYA][sharedCoordVecXA];
                            }

                            // Outer product computation
#pragma unroll
                            for (int iterVecSubTN = 0; iterVecSubTN < THREAD_SUBTILE_VEC_N; iterVecSubTN++) {
                                const int regCoordVecX = iterVecTN + iterVecSubTN;
#pragma unroll
                                for (int iterVecSubTK = 0; iterVecSubTK < THREAD_SUBTILE_VEC_K; iterVecSubTK++) {
                                    const int regBaseYB = iterVecSubTK * 4;

                                    regAccumulator[regCoordY][regCoordVecX].x += regA[iterVecSubTK].x * regB[
                                        regBaseYB + 0][iterVecSubTN].x;
                                    regAccumulator[regCoordY][regCoordVecX].y += regA[iterVecSubTK].x * regB[
                                        regBaseYB + 0][iterVecSubTN].y;
                                    regAccumulator[regCoordY][regCoordVecX].z += regA[iterVecSubTK].x * regB[
                                        regBaseYB + 0][iterVecSubTN].z;
                                    regAccumulator[regCoordY][regCoordVecX].w += regA[iterVecSubTK].x * regB[
                                        regBaseYB + 0][iterVecSubTN].w;

                                    regAccumulator[regCoordY][regCoordVecX].x += regA[iterVecSubTK].y * regB[
                                        regBaseYB + 1][iterVecSubTN].x;
                                    regAccumulator[regCoordY][regCoordVecX].y += regA[iterVecSubTK].y * regB[
                                        regBaseYB + 1][iterVecSubTN].y;
                                    regAccumulator[regCoordY][regCoordVecX].z += regA[iterVecSubTK].y * regB[
                                        regBaseYB + 1][iterVecSubTN].z;
                                    regAccumulator[regCoordY][regCoordVecX].w += regA[iterVecSubTK].y * regB[
                                        regBaseYB + 1][iterVecSubTN].w;

                                    regAccumulator[regCoordY][regCoordVecX].x += regA[iterVecSubTK].z * regB[
                                        regBaseYB + 2][iterVecSubTN].x;
                                    regAccumulator[regCoordY][regCoordVecX].y += regA[iterVecSubTK].z * regB[
                                        regBaseYB + 2][iterVecSubTN].y;
                                    regAccumulator[regCoordY][regCoordVecX].z += regA[iterVecSubTK].z * regB[
                                        regBaseYB + 2][iterVecSubTN].z;
                                    regAccumulator[regCoordY][regCoordVecX].w += regA[iterVecSubTK].z * regB[
                                        regBaseYB + 2][iterVecSubTN].w;

                                    regAccumulator[regCoordY][regCoordVecX].x += regA[iterVecSubTK].w * regB[
                                        regBaseYB + 3][iterVecSubTN].x;
                                    regAccumulator[regCoordY][regCoordVecX].y += regA[iterVecSubTK].w * regB[
                                        regBaseYB + 3][iterVecSubTN].y;
                                    regAccumulator[regCoordY][regCoordVecX].z += regA[iterVecSubTK].w * regB[
                                        regBaseYB + 3][iterVecSubTN].z;
                                    regAccumulator[regCoordY][regCoordVecX].w += regA[iterVecSubTK].w * regB[
                                        regBaseYB + 3][iterVecSubTN].w;
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store results to global memory
    const int globalBaseY = bidy * BLOCK_TILE_M + tidy;
    const int globalBaseVecX = bidx * BLOCK_TILE_VEC_N + tidx;

#pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_TILE_VEC_N; tn++) {
            const int globalCoordY = globalBaseY + tm * blockDim.y;
            const int globalCoordVecX = globalBaseVecX + tn * blockDim.x;
            const int dstIdx = globalCoordY * (N / 4) + globalCoordVecX;
            dstMat[dstIdx] = regAccumulator[tm][tn];
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


void runMySgemm(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 64;
    constexpr uint32_t BK = 16;
    constexpr uint32_t TM = 8;
    constexpr uint32_t TN = 8;
    constexpr uint32_t TK = 4;
    constexpr uint32_t STM = 4;
    constexpr uint32_t STN = 8;
    constexpr uint32_t STK = 4;
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

int main(int argc, char** argv) {
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaEvent_t evBegin, evEnd;
    cudaEventCreate(&evBegin);
    cudaEventCreate(&evEnd);

    // cuBLAS FLOPs ceiling is reached at 8192
    std::array SIZES{2048, 3072, 4096};

    int64_t m, n, k;
    int64_t max_size = SIZES.back();
    std::cout << "Max size: " << max_size << std::endl;

    float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;

    A = (float*)malloc(sizeof(float) * max_size * max_size);
    B = (float*)malloc(sizeof(float) * max_size * max_size);
    C = (float*)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float*)malloc(sizeof(float) * max_size * max_size);

    randomizeMat(A, max_size * max_size);
    randomizeMat(B, max_size * max_size);
    randomizeMat(C, max_size * max_size);

    CHECK_CUDA(cudaMalloc((void**)&dA, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void**)&dC, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void**)&dC_ref, sizeof(float) * max_size * max_size));

    CHECK_CUDA(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int64_t PERF_TIMES = 1;
    for (int size : SIZES) {
        m = n = k = size;
        runCublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);
        runMySgemm(m, n, k, alpha, dA, dB, beta, dC);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError()); // Check for async errors during kernel run
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (!verifyMat(C_ref, C, m * n)) {
            std::cout << "verification failed" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<float> elapsedTimes;
        for (int j = 0; j < PERF_TIMES; j++) {
            cudaEventRecord(evBegin);
            runMySgemm(m, n, k, alpha, dA, dB, beta, dC);
            cudaEventRecord(evEnd);
            cudaEventSynchronize(evBegin);
            cudaEventSynchronize(evEnd);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, evBegin, evEnd);
            elapsedTimes.push_back(elapsedTime);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)m * n * k * 2;
        const float meanTflops = macs / meanTime / 1e9;

        std::cout << "=============================" << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "Performance: " << meanTflops << " tflops" << std::endl;
    }

    // Free up CPU and GPU space
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);

    return 0;
};