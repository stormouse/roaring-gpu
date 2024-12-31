#include <cuda_common.cuh>
#include <iostream>
#include <roaring.cuh>
#include <roaring_helper.cuh>

using namespace tora::roaring;

int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (2048 * 1024 * 1024UL));

    RoaringBitmapDevice bitmap1 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice bitmap2 = getRandomRoaringBitmap(0, 1024, 512, 512, 1, 2048);
    RoaringBitmapDevice result = getIntermediateBitmap(0, 1024);

    int threadsPerBlock = 64;
    int blocksPerGrid = 16;
    int sharedMemBytes = threadsPerBlock * sizeof(uint32_t);

    bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(
        *bitmap1.devPtr(), *bitmap2.devPtr(), *result.devPtr(), 0, 1024);

    uint32_t cards[3];
    uint32_t* cards_device;
    checkCuda(cudaMalloc((void**)&cards_device, sizeof(uint32_t) * 3));
    checkCuda(cudaMemset(cards_device, 0, sizeof(uint32_t) * 3));

    bitmapGetCardinality<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(bitmap1.dev(), &cards_device[0], 0, 1024);
    cudaDeviceSynchronize(); std::cout << std::endl;
    bitmapGetCardinality<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(bitmap2.dev(), &cards_device[1], 0, 1024);
    cudaDeviceSynchronize(); std::cout << std::endl;
    bitmapGetCardinality<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(result.dev(), &cards_device[2], 0, 1024);
    cudaDeviceSynchronize(); std::cout << std::endl;

    checkCuda(cudaMemcpy(cards, cards_device, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    std::cout << "cardinality(b1) = " << cards[0] << "\n";
    std::cout << "cardinality(b2) = " << cards[1] << "\n";
    std::cout << "cardinality(b3) = " << cards[2] << std::endl;

    return 0;
}