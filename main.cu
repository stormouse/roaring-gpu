#include "bitmap_example.cuh"

int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (1024 * 1024 * 1024UL));

    testBitmapIntersect();

    cudaDeviceSynchronize();

    testBitmapUnion();

    return 0;
}