#pragma once

#include <stdio.h>

namespace tora
{

__host__ __device__ inline void* custom_malloc(size_t size)
{
    #if DEBUG
    void* p = malloc(size);
    printf("malloc: %p\n", p);
    return p;
    #else
    return malloc(size);
    #endif
}

__host__ __device__ inline void custom_free(void* ptr)
{
    #if DEBUG
    printf("free: %p\n", ptr);
    #endif

    free(ptr);
}

} // namespace tora