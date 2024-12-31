#include <gtest/gtest.h>
#include <cuda_common.cuh>
#include <memory.cuh>
#include <random>
#include <roaring.cuh>
#include <roaring_helper.cuh>

using namespace tora::roaring;

__global__ void printContainerInfo(const RoaringBitmapFlat& bitmap, int containerIndex)
{
    printf(
        "index %d, type=%d, cardinality=%d, capacity=%d, data=%p\n", containerIndex,
        bitmap.containers[containerIndex].type, bitmap.containers[containerIndex].cardinality,
        bitmap.containers[containerIndex].capacity, bitmap.containers[containerIndex].data);
}

TEST(RoaringGpu, BitmapUnion)
{
    RoaringBitmapDevice bitmap1;
    RoaringBitmapDevice bitmap2;
    RoaringBitmapDevice result = getIntermediateBitmap(0, 8);

    for (int i = 0; i < 8; i++)
    {
        bitmap1.setBit(66017 + i, true);
        bitmap2.setBit(66023 + i, true);
    }

    printContainerInfo<<<1, 1>>>(*bitmap1.devPtr(), 1);
    printContainerInfo<<<1, 1>>>(*bitmap2.devPtr(), 1);

    int threadsPerBlock = 16;
    int blocksPerGrid = 16;

    bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 8);

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 1);

    cudaDeviceSynchronize();
}

TEST(RoaringGpu, BitmapIntersect)
{
    RoaringBitmapDevice bitmap1;
    RoaringBitmapDevice bitmap2;
    RoaringBitmapDevice result = getIntermediateBitmap(0, 8);

    for (int i = 0; i < 8; i++)
    {
        bitmap1.setBit(66017 + i, true);
        bitmap2.setBit(66023 + i, true);
    }

    printContainerInfo<<<1, 1>>>(*bitmap1.devPtr(), 1);
    printContainerInfo<<<1, 1>>>(*bitmap2.devPtr(), 1);

    int threadsPerBlock = 64;
    int blocksPerGrid = 16;

    bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 8);

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 1);

    cudaDeviceSynchronize();
}

TEST(RoaringGpu, RandomBitmapUnion)
{
    RoaringBitmapDevice bitmap1 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice bitmap2 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice result = getIntermediateBitmap(0, 1024);

    printContainerInfo<<<1, 1>>>(*bitmap1.devPtr(), 18);
    printContainerInfo<<<1, 1>>>(*bitmap2.devPtr(), 18);

    int threadsPerBlock = 64;
    int blocksPerGrid = 16;

    bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 1024);

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 18);

    cudaDeviceSynchronize();
}

TEST(RoaringGpu, RandomBitmapIntersect)
{
    RoaringBitmapDevice bitmap1 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice bitmap2 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice result = getIntermediateBitmap(0, 1024);

    printContainerInfo<<<1, 1>>>(*bitmap1.devPtr(), 18);
    printContainerInfo<<<1, 1>>>(*bitmap2.devPtr(), 18);

    int threadsPerBlock = 64;
    int blocksPerGrid = 16;

    bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 1024);

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 18);

    cudaDeviceSynchronize();
}

TEST(RoaringGpu, RandomBitmapGetCardinality)
{
    RoaringBitmapDevice bitmap1 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice bitmap2 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice result = getIntermediateBitmap(0, 1024);

    int threadsPerBlock = 64;
    int blocksPerGrid = 16;
    int sharedMemBytes = threadsPerBlock * sizeof(uint32_t);

    bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 1024);

    constexpr int numResults = 4;
    uint32_t cards[numResults];
    uint32_t* cards_device;
    checkCuda(cudaMalloc((void**)&cards_device, sizeof(uint32_t) * numResults));
    checkCuda(cudaMemset(cards_device, 0, sizeof(uint32_t) * numResults));

    bitmapGetCardinality<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(bitmap1.dev(), &cards_device[0], 0, 1024);
    bitmapGetCardinality<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(bitmap2.dev(), &cards_device[1], 0, 1024);
    bitmapGetCardinality<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(result.dev(), &cards_device[2], 0, 1024);
    cudaDeviceSynchronize();

    bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 1024);
    bitmapGetCardinality<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(result.dev(), &cards_device[3], 0, 1024);
    cudaDeviceSynchronize();
    
    checkCuda(cudaMemcpy(cards, cards_device, sizeof(uint32_t) * numResults, cudaMemcpyDeviceToHost));

    EXPECT_LE(cards[2], min(cards[0], cards[1]));
    EXPECT_GE(cards[3], max(cards[0], cards[1]));

    std::cerr << "cardinality(b1) = " << cards[0] << "\n";
    std::cerr << "cardinality(b2) = " << cards[1] << "\n";
    std::cerr << "cardinality(b1 & b2) = " << cards[2] << "\n";
    std::cerr << "cardinality(b1 | b2) = " << cards[3] << std::endl;
}

struct DeviceObject
{
    char* deviceData;
};

__global__ void createString1(DeviceObject& devObj)
{
    devObj.deviceData = (char*)tora::custom_malloc(8 * sizeof(char));
    devObj.deviceData[0] = 'h';
    devObj.deviceData[1] = 'e';
    devObj.deviceData[2] = 'l';
    devObj.deviceData[3] = 'l';
    devObj.deviceData[4] = 'o';
    devObj.deviceData[5] = '\0';
}

__global__ void createString2(DeviceObject& devObj)
{
    devObj.deviceData = (char*)tora::custom_malloc(8 * sizeof(char));
    devObj.deviceData[0] = 'w';
    devObj.deviceData[1] = 'o';
    devObj.deviceData[2] = 'r';
    devObj.deviceData[3] = 'l';
    devObj.deviceData[4] = 'd';
    devObj.deviceData[5] = '\0';
}

__global__ void printString(const DeviceObject& devObj)
{
    printf("%s\n", devObj.deviceData);
}

__global__ void freeString(DeviceObject& devObj)
{
    tora::custom_free(devObj.deviceData);
}

void malloc_test1()
{
    DeviceObject* devObj;
    checkCuda(cudaMalloc((void**)&devObj, sizeof(DeviceObject)));

    createString1<<<1, 1>>>(*devObj);
    printString<<<1, 1>>>(*devObj);
    freeString<<<1, 1>>>(*devObj);
}

void malloc_test2()
{
    DeviceObject* devObj;
    checkCuda(cudaMalloc((void**)&devObj, sizeof(DeviceObject)));

    createString2<<<1, 1>>>(*devObj);
    printString<<<1, 1>>>(*devObj);
    freeString<<<1, 1>>>(*devObj);
}