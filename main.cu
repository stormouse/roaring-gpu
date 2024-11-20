#include "bitmap_example.cuh"
#include <iostream>

int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (2048 * 1024 * 1024UL));

    // malloc_test1();
    // cudaDeviceSynchronize();
    // malloc_test2();
    // cudaDeviceSynchronize();

    testBitmapUnion();

    cudaDeviceSynchronize();

    std::cout << "\n----------------dividing-line----------------\n" << "\n";

    testBitmapIntersect();
    
    cudaDeviceSynchronize();

    return 0;
}