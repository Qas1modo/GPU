#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32
#define FULL_MASK 0xFFFFFFFF
#define TILE_SIZE_HALF 16


__global__ void calculateAccount(int *changes, int* account, int* sum, int clients, int periods) {
    __shared__ int prevBlockData[TILE_SIZE][TILE_SIZE + 1];
    __shared__ int blockData[TILE_SIZE][TILE_SIZE + 1];
    __shared__ int nextBlockData[TILE_SIZE][TILE_SIZE + 1];
    int* prevBlock = &prevBlockData[0][0];
    int* block = &blockData[0][0];
    int* nextBlock = &nextBlockData[0][0];
    int threadSum = 0;
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tile_index = (threadIdx.y % TILE_SIZE_HALF) * (TILE_SIZE + 1) + threadIdx.x;
    int in_index = threadIdx.y * clients + x;
    int index_out = (threadIdx.y % TILE_SIZE_HALF) * clients + x;
    prevBlockData[threadIdx.y][threadIdx.x] = changes[in_index];
    in_index += TILE_SIZE * clients;
    __syncthreads();
    if (threadIdx.y == TILE_SIZE_HALF) {       
        for (int threadRow = 0; threadRow < TILE_SIZE; threadRow++) {
            threadSum += prevBlock[threadRow * (TILE_SIZE + 1) + threadIdx.x];
            prevBlock[threadRow * (TILE_SIZE + 1) + threadIdx.x] = threadSum;
        }
    }
    blockData[threadIdx.y][threadIdx.x] = changes[in_index];
    in_index += TILE_SIZE * clients;
    __syncthreads();
    for (int block_Y = 2 * TILE_SIZE; block_Y < periods; block_Y+=TILE_SIZE) {
        if (threadIdx.y < TILE_SIZE_HALF) {
            nextBlock[tile_index] = changes[in_index];
            in_index += TILE_SIZE_HALF * clients;
            nextBlock[tile_index + (TILE_SIZE_HALF) * (TILE_SIZE + 1)] = changes[in_index];
            in_index += TILE_SIZE_HALF * clients;
        } else {
            if (threadIdx.y == TILE_SIZE_HALF) {
                for (int threadRow = 0; threadRow < TILE_SIZE * (TILE_SIZE + 1); threadRow += TILE_SIZE + 1) {
                    threadSum += block[threadRow + threadIdx.x];
                    block[threadRow + threadIdx.x] = threadSum;
                }
            }
            int threadRowSum = prevBlock[tile_index];
            int y = (index_out - x) / clients;
            account[index_out] = threadRowSum;
            for (int offset = warpSize/2; offset > 0; offset /= 2) threadRowSum += __shfl_down_sync(FULL_MASK, threadRowSum, offset);
            if (threadIdx.x == 0) {
                atomicAdd(&sum[y], threadRowSum);
            }
            index_out += TILE_SIZE_HALF * clients; 
            threadRowSum = prevBlock[tile_index + TILE_SIZE_HALF * (TILE_SIZE + 1)];
            account[index_out] = threadRowSum;
            for (int offset = warpSize/2; offset > 0; offset /= 2) threadRowSum += __shfl_down_sync(FULL_MASK, threadRowSum, offset);
            if (threadIdx.x == 0) {
                atomicAdd(&sum[y + TILE_SIZE_HALF], threadRowSum);
            }
            index_out += TILE_SIZE_HALF * clients; 
        }
        __syncthreads();
        int* temp = prevBlock;
        prevBlock = block;
        block = nextBlock;
        nextBlock = temp;
    }
    if (threadIdx.y == TILE_SIZE_HALF) {
        for (int threadRow = 0; threadRow < TILE_SIZE * (TILE_SIZE + 1); threadRow += TILE_SIZE + 1) {
            threadSum += block[threadRow + threadIdx.x];
            block[threadRow + threadIdx.x] = threadSum;
        }
    }
    tile_index = threadIdx.y * (TILE_SIZE + 1) + threadIdx.x;
    threadSum = prevBlock[tile_index];
    account[(threadIdx.y + periods - 2 * TILE_SIZE) * clients + x] = threadSum;
    for (int offset = warpSize/2; offset > 0; offset /= 2) threadSum += __shfl_down_sync(FULL_MASK, threadSum, offset);
    if (threadIdx.x == 0) {
        atomicAdd(&sum[periods - 2 *TILE_SIZE + threadIdx.y], threadSum);
    }
    __syncthreads();
    threadSum = block[tile_index];
    account[(threadIdx.y + periods - TILE_SIZE) * clients + x] = threadSum;
    for (int offset = warpSize/2; offset > 0; offset /= 2) threadSum += __shfl_down_sync(FULL_MASK, threadSum, offset);
    if (threadIdx.x == 0) {
        atomicAdd(&sum[periods - TILE_SIZE + threadIdx.y], threadSum);
    }
}


void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    dim3 numThreads(TILE_SIZE, TILE_SIZE);
    calculateAccount<<<clients / TILE_SIZE, numThreads>>>(changes, account, sum, clients, periods);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

