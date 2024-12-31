#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <vector>
#include "bitop.cuh"
#include "cuda_common.cuh"
#include "memory.cuh"
#include "roaring_helper.cuh"

namespace tora::roaring
{

__global__ void initializeIntermediateResultContainers(RoaringBitmapFlat* bitmap, int containerLow, int containerHigh)
{
    int n = containerHigh - containerLow;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    while (idx < n)
    {
        bitmap->containers[containerLow + idx].data = (uint32_t*)custom_malloc(65536);
        bitmap->containers[containerLow + idx].capacity = 8192;
        bitmap->containers[containerLow + idx].cardinality = 0;
        idx += step;
    }
}

__global__ void buildRandomArrayContainers(
    RoaringBitmapFlat* bitmap, int* containerIndexes, int n, int arrayElementLow, int arrayElementHigh)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx < n)
    {
        curandState state;
        curand_init(42, idx, 0, &state);
        float randomValue = curand_uniform(&state);

        int numElements = arrayElementLow + static_cast<int>((arrayElementHigh - arrayElementLow) * randomValue);

        Container& dst = bitmap->containers[containerIndexes[idx]];
        dst.capacity = (numElements + 1) / 2;
        dst.data = (uint32_t*)custom_malloc(dst.capacity * sizeof(uint32_t));
        dst.type = ContainerType::Array;
        dst.cardinality = 0;
        
        for (int i = 0; i < dst.capacity; i++)
        {
            dst.data[i] = 0;
        }

        for (int i = 0; i < numElements; i++)
        {
            array_setBit(dst, i, true);
        }

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void buildRandomBitsetContainers(RoaringBitmapFlat* bitmap, int* containerIndexes, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx < n)
    {
        curandState state;
        curand_init(42, idx, 0, &state);

        Container& dst = bitmap->containers[containerIndexes[idx]];
        dst.capacity = 2048;
        dst.data = (uint32_t*)custom_malloc(dst.capacity * sizeof(uint32_t));
        dst.type = ContainerType::Bitset;
        dst.cardinality = 0;

        for (int i = 0; i < dst.capacity; i++)
        {
            uint32_t value = (uint32_t)(curand_uniform(&state) * (uint64_t)(UINT_MAX));
            dst.data[i] = value;
            dst.cardinality += bitsSet(value);
        }

        idx += gridDim.x * blockDim.x;
    }
}

RoaringBitmapDevice getRandomRoaringBitmap(
    int containerLow, int containerHigh, int numArrays, int numBitsets, int arrayElementLow, int arrayElementHigh)
{
    static const int ARRAY = 0;
    static const int BITSET = 2;

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> containerIndex(containerHigh - containerLow);
    std::vector<int> containerType(numArrays + numBitsets);

    for (int i = 0; i < containerHigh - containerLow; i++)
    {
        containerIndex[i] = containerLow + i;
    }

    for (int i = 0; i < numArrays; i++)
    {
        containerType[i] = ARRAY;
    }

    for (int i = numArrays; i < numArrays + numBitsets; i++)
    {
        containerType[i] = BITSET;
    }

    std::shuffle(containerIndex.begin(), containerIndex.end(), g);
    std::shuffle(containerType.begin(), containerType.end(), g);

    int numContainers = std::min(containerHigh - containerLow, numArrays + numBitsets);

    std::vector<int> arrayIndexes;
    std::vector<int> bitsetIndexes;

    for (int i = 0; i < numContainers; i++)
    {
        if (containerType[i] == ARRAY)
        {
            arrayIndexes.push_back(containerIndex[i]);
        }
        else
        {
            bitsetIndexes.push_back(containerIndex[i]);
        }
    }

    int* arrayIndexes_d;
    int* bitsetIndexes_d;
    checkCuda(cudaMalloc((void**)&arrayIndexes_d, sizeof(int) * numArrays));
    checkCuda(cudaMalloc((void**)&bitsetIndexes_d, sizeof(int) * numBitsets));
    checkCuda(
        cudaMemcpy(arrayIndexes_d, arrayIndexes.data(), sizeof(int) * arrayIndexes.size(), cudaMemcpyHostToDevice));
    checkCuda(
        cudaMemcpy(bitsetIndexes_d, bitsetIndexes.data(), sizeof(int) * bitsetIndexes.size(), cudaMemcpyHostToDevice));

    RoaringBitmapDevice bitmap;

    int threadsPerBlock = 256;
    int blocksPerGrid = 64;

    buildRandomArrayContainers<<<blocksPerGrid, threadsPerBlock>>>(
        bitmap.devPtr(), arrayIndexes_d, arrayIndexes.size(), arrayElementLow, arrayElementHigh);
    buildRandomBitsetContainers<<<blocksPerGrid, threadsPerBlock>>>(
        bitmap.devPtr(), bitsetIndexes_d, bitsetIndexes.size());

    checkCuda(cudaFree(arrayIndexes_d));
    checkCuda(cudaFree(bitsetIndexes_d));

    cudaDeviceSynchronize();

    return bitmap;
}


RoaringBitmapDevice getIntermediateBitmap(int containerLow, int containerHigh)
{
    RoaringBitmapDevice bitmap;
    initializeIntermediateResultContainers<<<256, 64>>>(bitmap.devPtr(), containerLow, containerHigh);
    return bitmap;
}

}  // namespace tora::roaring