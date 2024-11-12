#pragma once

#include <cstdint>

namespace tora::roaring
{

enum class ContainerType
{
    Bitset = 1,
    Array = 2,
    Run = 3,
};

struct Container
{
    ContainerType type;
    uint32_t cardinality = 0;
    uint32_t capacity = 0;  // capacity in number of uint32
    uint32_t* data = nullptr;
};

__host__ __device__ Container bitset_bitset_union(const Container& c1, const Container& c2);
__host__ __device__ Container bitset_bitset_intersect(const Container& c1, const Container& c2);
__host__ __device__ Container array_bitset_union(const Container& c1, const Container& c2);
__host__ __device__ Container array_bitset_intersect(const Container& c1, const Container& c2);
__host__ __device__ Container array_array_union(const Container& c1, const Container& c2);
__host__ __device__ Container array_array_intersect(const Container& c1, const Container& c2);

}  // namespace tora::roaring