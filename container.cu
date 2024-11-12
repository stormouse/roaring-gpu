#include "bitop.cuh"
#include "container.cuh"

namespace tora::roaring
{

__host__ __device__ Container bitset_bitset_union(const Container& c1, const Container& c2)
{
    Container dst;
    int minLen = c1.capacity < c2.capacity ? c1.capacity : c2.capacity;
    int maxLen = c1.capacity + c2.capacity - minLen;

#if defined(__CUDA_ARCH__)
    cudaMalloc((void**)&(dst.data), maxLen * sizeof(uint32_t));
#else
    dst.data = (uint32_t*)malloc(maxLen * sizeof(uint32_t));
#endif

    dst.type = ContainerType::Bitset;
    dst.capacity = maxLen;

    for (int i = 0; i < minLen; i++)
    {
        dst.data[i] = c1.data[i] | c2.data[i];
    }

    const Container* r = c1.capacity > minLen ? &c1 : &c2;
    for (int i = minLen; i < maxLen; i++)
    {
        dst.data[i] = r->data[i];
    }

    return dst;
}

__host__ __device__ Container bitset_bitset_intersect(const Container& c1, const Container& c2)
{
    Container dst;
    int minLen = c1.capacity < c2.capacity ? c1.capacity : c2.capacity;

#if defined(__CUDA_ARCH__)
    cudaMalloc((void**)&(dst.data), minLen * sizeof(uint32_t));
#else
    dst.data = (uint32_t*)malloc(minLen * sizeof(uint32_t));
#endif

    dst.type = ContainerType::Bitset;
    dst.capacity = minLen;

    for (int i = 0; i < minLen; i++)
    {
        dst.data[i] = c1.data[i] & c2.data[i];
    }

    // TODO: convert it to array container if cardinality < 4K.

    return dst;
}

__host__ __device__ Container array_bitset_union(const Container& c1, const Container& c2)
{
    const Container& arr = c1.type == ContainerType::Array ? c1 : c2;
    const Container& bitset = c1.type == ContainerType::Bitset ? c1 : c2;

    Container dst;
    uint16_t* arrayElements = (uint16_t*)arr.data;
    int requiredCapacity = bitset.capacity;
    if (requiredCapacity * sizeof(uint32_t) < arrayElements[arr.cardinality - 1])
    {
        requiredCapacity = (arrayElements[arr.cardinality - 1] + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    }

#if defined(__CUDA_ARCH__)
    cudaMalloc((void**)&(dst.data), requiredCapacity * sizeof(uint32_t));
#else
    dst.data = (uint32_t*)malloc(requiredCapacity * sizeof(uint32_t));
#endif

#if defined(__CUDA_ARCH__)
    for (int i = 0; i < bitset.capacity; i++)
    {
        dst.data[i] = bitset.data[i];
    }
#else
    memcpy(dst.data, bitset.data, bitset.capacity * sizeof(uint32_t));
#endif

    dst.type = ContainerType::Bitset;
    dst.capacity = requiredCapacity;

    int j = 0;
    for (int i = 0; i < arr.cardinality; i++)
    {
        uint16_t element = arrayElements[i];
        int offset = element >> 5;
        int bitpos = element & 31;
        dst.data[offset] |= 1 << bitpos;
        j += 1 - ((bitset.data[offset] & (1 << bitpos)) > 0);
    }

    dst.cardinality = bitset.cardinality + j;

    return dst;
}

__host__ __device__ Container array_bitset_intersect(const Container& c1, const Container& c2)
{
    const Container& arr = c1.type == ContainerType::Array ? c1 : c2;
    const Container& bitset = c1.type == ContainerType::Bitset ? c1 : c2;

    Container dst;
    uint16_t* arrayElements = (uint16_t*)arr.data;
    int requiredCapacity = arr.capacity;

#if defined(__CUDA_ARCH__)
    cudaMalloc((void**)&(dst.data), requiredCapacity * sizeof(uint32_t));
#else
    dst.data = (uint32_t*)malloc(requiredCapacity * sizeof(uint32_t));
#endif

    dst.type = ContainerType::Array;
    dst.capacity = requiredCapacity;
    uint16_t* dstElements = (uint16_t*)dst.data;

    int j = 0;
    for (int i = 0; i < arr.cardinality; i++)
    {
        uint16_t element = arrayElements[i];
        int offset = element >> 5;
        int bitpos = element & 31;
        if (bitset.data[offset] & (1 << bitpos))
        {
            dstElements[j++] = element;
        }
    }

    dst.cardinality = j;

    return dst;
}

__host__ __device__ Container array_array_union(const Container& c1, const Container& c2)
{
    Container dst;
    uint16_t* a1 = (uint16_t*)c1.data;
    uint16_t* a2 = (uint16_t*)c2.data;
    int requiredCapacity = (c1.cardinality + c2.cardinality) * 2;  // `sizeof(uint32_t) / sizeof(uint16_t)`

#if defined(__CUDA_ARCH__)
    cudaMalloc((void**)&(dst.data), requiredCapacity * sizeof(uint32_t));
#else
    dst.data = (uint32_t*)malloc(requiredCapacity * sizeof(uint32_t));
#endif

    dst.type = ContainerType::Array;
    dst.capacity = requiredCapacity;

    uint16_t* dstElements = (uint16_t*)dst.data;
    int i = 0, j = 0, k = 0;
    while (i < c1.cardinality && j < c2.cardinality)
    {
        if (a1[i] == a2[j])
        {
            dstElements[k++] = a1[i];
            i++;
            j++;
        }
        else if (a1[i] < a2[j])
        {
            dstElements[k++] = a1[i++];
        }
        else
        {
            dstElements[k++] = a2[j++];
        }
    }

    while (i < c1.cardinality)
    {
        dstElements[k++] = a1[i++];
    }

    while (j < c2.cardinality)
    {
        dstElements[k++] = a2[j++];
    }

    dst.cardinality = k;

    // TODO: convert to bitset when cardinality is high

    return dst;
}

__host__ __device__ Container array_array_intersect(const Container& c1, const Container& c2)
{
    Container dst;
    uint16_t* a1 = (uint16_t*)c1.data;
    uint16_t* a2 = (uint16_t*)c2.data;
    int requiredCapacity = (c1.cardinality < c2.cardinality ? c1.cardinality : c2.cardinality) *
                           2;  // `sizeof(uint32_t) / sizeof(uint16_t)`

#if defined(__CUDA_ARCH__)
    cudaMalloc((void**)&(dst.data), requiredCapacity * sizeof(uint32_t));
#else
    dst.data = (uint32_t*)malloc(requiredCapacity * sizeof(uint32_t));
#endif

    dst.type = ContainerType::Array;
    dst.capacity = requiredCapacity;

    uint16_t* dstElements = (uint16_t*)dst.data;
    int i = 0, j = 0, k = 0;
    while (i < c1.cardinality && j < c2.cardinality)
    {
        if (a1[i] == a2[j])
        {
            dstElements[k++] = a1[i];
            i++;
            j++;
        }
        else if (a1[i] < a2[j])
        {
            a1[i++];
        }
        else
        {
            a2[j++];
        }
    }

    dst.cardinality = k;

    return dst;
}

}  // namespace tora::roaring