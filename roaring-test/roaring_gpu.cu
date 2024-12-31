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

void testBitmapUnion()
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

    bitmapUnionInplace<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 8);

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 1);

    cudaDeviceSynchronize();
}

TEST(RoaringGpu, BitmapUnion)
{
    testBitmapUnion();
}

void testBitmapIntersect()
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

    bitmapIntersectInplace<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 8);

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 1);

    cudaDeviceSynchronize();
}

TEST(RoaringGpu, BitmapIntersect)
{
    testBitmapIntersect();
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