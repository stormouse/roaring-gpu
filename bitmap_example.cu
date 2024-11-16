#include <iostream>
#include "cuda_common.cuh"
#include "roaring.cuh"

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
    static const int BitmapFlatSize = 65536;

    RoaringBitmapDevice bitmap1;
    RoaringBitmapDevice bitmap2;
    RoaringBitmapDevice result;

    for (int i = 0; i < 8; i++)
    {
        bitmapSetBit<<<1, 1>>>(*bitmap1.devPtr(), 66017 + i, true);
        bitmapSetBit<<<1, 1>>>(*bitmap2.devPtr(), 66023 + i, true);
    }

    printContainerInfo<<<1, 1>>>(*bitmap1.devPtr(), 1);
    printContainerInfo<<<1, 1>>>(*bitmap2.devPtr(), 1);

    int threadsPerBlock = 256;
    int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;

    bitmapUnion<<<blocksPerGrid, threadsPerBlock>>>(*bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr());

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 1);

    cudaDeviceSynchronize();

    // bool* outputValue;
    // checkCuda(cudaMallocHost((void**)&outputValue, sizeof(bool)));
    // for (int i = 66014; i <= 66037; i++)
    // {
    //     bitmapGetBit<<<1, 1>>>(*result.devPtr(), i, outputValue);
    //     cudaDeviceSynchronize();
    //     std::cout << i << ": " << *outputValue << "\n";
    // }
}

void testBitmapIntersect()
{
    static const int BitmapFlatSize = 65536;

    RoaringBitmapDevice bitmap1;
    RoaringBitmapDevice bitmap2;
    RoaringBitmapDevice result;

    for (int i = 0; i < 8; i++)
    {
        bitmapSetBit<<<1, 1>>>(*bitmap1.devPtr(), 66017 + i, true);
        bitmapSetBit<<<1, 1>>>(*bitmap2.devPtr(), 66023 + i, true);
    }

    printContainerInfo<<<1, 1>>>(*bitmap1.devPtr(), 1);
    printContainerInfo<<<1, 1>>>(*bitmap2.devPtr(), 1);

    int threadsPerBlock = 256;
    int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;

    bitmapIntersect<<<blocksPerGrid, threadsPerBlock>>>(*bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr());

    printContainerInfo<<<1, 1>>>(*result.devPtr(), 1);

    cudaDeviceSynchronize();

    // bool* outputValue;
    // checkCuda(cudaMallocHost((void**)&outputValue, sizeof(bool)));
    // for (int i = 66014; i <= 66037; i++)
    // {
    //     bitmapGetBit<<<1, 1>>>(*result.devPtr(), i, outputValue);
    //     cudaDeviceSynchronize();
    //     std::cout << i << ": " << *outputValue << "\n";
    // }
}