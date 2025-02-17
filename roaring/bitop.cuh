#pragma once

#include <cstdint>

namespace tora::roaring
{

// Note: trailingZeros returns 0 when v == 0
__host__ __device__ inline int trailingZeros(uint32_t v)
{
    static const int MultiplyDeBruijnBitPosition[32] = {0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
                                                        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};
    return MultiplyDeBruijnBitPosition[((uint32_t)((v & -v) * 0x077CB531U)) >> 27];
}

__host__ __device__ inline int bitsSet(uint32_t v)
{
    int c;
    for (c = 0; v; c++)
    {
        v &= v - 1;
    }
    return c;
}

}  // namespace tora::roaring