#pragma once

namespace tora
{

__host__ __device__ inline void* custom_malloc(size_t size)
{
    return malloc(size);
}

} // namespace tora