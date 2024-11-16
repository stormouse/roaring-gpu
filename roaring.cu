#include <stdio.h>
#include "cuda_common.cuh"
#include "roaring.cuh"

namespace tora::roaring
{

static const int BitmapFlatSize = 65536;

RoaringBitmapDevice::RoaringBitmapDevice()
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;
    checkCuda(cudaMalloc((void**)&deviceData_, sizeof(RoaringBitmapFlat)));
    allocateFlatContainers<<<1, 1>>>(*deviceData_);
    initBitmapContainers<<<blocksPerGrid, threadsPerBlock>>>(deviceData_);
}

RoaringBitmapDevice::RoaringBitmapDevice(int stream)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;
    checkCuda(cudaMalloc((void**)&deviceData_, sizeof(RoaringBitmapFlat)));
    allocateFlatContainers<<<1, 1>>>(*deviceData_);
    initBitmapContainers<<<blocksPerGrid, threadsPerBlock, stream>>>(deviceData_);
}

RoaringBitmapDevice::~RoaringBitmapDevice()
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;
    freeBitmapContainers<<<blocksPerGrid, threadsPerBlock>>>(deviceData_);
    freeFlatContainers<<<1, 1>>>(deviceData_);
    checkCuda(cudaFree(deviceData_));
}

__global__ void allocateFlatContainers(RoaringBitmapFlat& a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        a.containers = (Container*)malloc(65536 * sizeof(Container));
        printf("a.containers at %p\n", a.containers);
    }
}

__global__ void initBitmapContainers(RoaringBitmapFlat* bitmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        bitmap->containers[idx].data = nullptr;
        bitmap->containers[idx].type = ContainerType::Array;
        bitmap->containers[idx].cardinality = 0;
        bitmap->containers[idx].capacity = 0;
    }
}

__global__ void freeBitmapContainers(RoaringBitmapFlat* bitmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        if (bitmap->containers[idx].data != nullptr)
        {
            printf("free: bitmap->containers[%d].data at %p\n", idx, bitmap->containers[idx].data);
            free(bitmap->containers[idx].data);
        }
    }
}

__global__ void freeFlatContainers(RoaringBitmapFlat* a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        printf("free: a.containers at %p\n", a->containers);
        free(a->containers);
    }
}

__host__ __device__ inline constexpr int typePair(ContainerType a, ContainerType b)
{
    return (1 << (int)a) + (1 << (int)b);
}

__global__ void bitmapUnion(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        const Container& c1 = a.containers[idx];
        const Container& c2 = b.containers[idx];
        switch (typePair(c1.type, c2.type))
        {
            case typePair(ContainerType::Bitset, ContainerType::Bitset):
            {
                t.containers[idx] = bitset_bitset_union(c1, c2);
                if (t.containers[idx].cardinality > 0 || c1.cardinality > 0 || c2.cardinality > 0)
                {
                    printf(
                        "debug: bitset_bitset_union at index %d, c1.type=%d, c2.type=%d, c1.cardinality=%d, "
                        "c2.cardinality=%d, t.containers[idx]={card=%d, capa=%d}\n",
                        idx, c1.type, c2.type, c1.cardinality, c2.cardinality, t.containers[idx].cardinality,
                        t.containers[idx].capacity);
                }
            }
            break;

            case typePair(ContainerType::Bitset, ContainerType::Array):
            {
                t.containers[idx] = array_bitset_union(c1, c2);
                if (t.containers[idx].cardinality > 0 || c1.cardinality > 0 || c2.cardinality > 0)
                {
                    printf(
                        "debug: array_bitset_union at index %d, c1.type=%d, c2.type=%d, c1.cardinality=%d, "
                        "c2.cardinality=%d, t.containers[idx]={card=%d, capa=%d}\n",
                        idx, c1.type, c2.type, c1.cardinality, c2.cardinality, t.containers[idx].cardinality,
                        t.containers[idx].capacity);
                }
            }
            break;

            case typePair(ContainerType::Array, ContainerType::Array):
            {
                t.containers[idx] = array_array_union(c1, c2);
                if (t.containers[idx].cardinality > 0 || c1.cardinality > 0 || c2.cardinality > 0)
                {
                    printf(
                        "debug: array_array_union at index %d, c1.type=%d, c2.type=%d, c1.cardinality=%d, "
                        "c2.cardinality=%d, t.containers[idx]={card=%d, capa=%d}\n",
                        idx, c1.type, c2.type, c1.cardinality, c2.cardinality, t.containers[idx].cardinality,
                        t.containers[idx].capacity);
                }
            }
            break;
        }
    }
}

__global__ void bitmapIntersect(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        const Container& c1 = a.containers[idx];
        const Container& c2 = b.containers[idx];
        switch (typePair(c1.type, c2.type))
        {
            case typePair(ContainerType::Bitset, ContainerType::Bitset):
            {
                t.containers[idx] = bitset_bitset_intersect(c1, c2);
            }
            break;

            case typePair(ContainerType::Bitset, ContainerType::Array):
            {
                t.containers[idx] = array_bitset_intersect(c1, c2);
            }
            break;

            case typePair(ContainerType::Array, ContainerType::Array):
            {
                t.containers[idx] = array_array_intersect(c1, c2);
            }
            break;
        }
    }
}

__global__ void bitmapGetBit(const RoaringBitmapFlat& a, int pos, bool* outValue)
{
    if (threadIdx.x == 0)
    {
        *outValue = a.getBit(pos);
    }
}

__global__ void bitmapSetBit(RoaringBitmapFlat& a, int pos, bool value)
{
    if (threadIdx.x == 0)
    {
        a.setBit(pos, value);
    }
}

__host__ __device__ bool RoaringBitmapFlat::getBit(int pos) const
{
    int containerIndex = (pos >> 16);
    int offset = pos & 65535;
    if (containers[containerIndex].cardinality == 0)
    {
        return false;
    }
    else
    {
        const Container& c = containers[containerIndex];
        switch (c.type)
        {
            case ContainerType::Bitset:
                return bitset_getBit(c, offset);
                break;

            case ContainerType::Array:
                return array_getBit(c, offset);
                break;
        }
    }

    return false;
}

__host__ __device__ void RoaringBitmapFlat::setBit(int pos, bool value)
{
    int containerIndex = (pos >> 16);
    int offset = pos & 65535;
    if (containers[containerIndex].data == nullptr)
    {
        // TODO: save me from my laziness
        containers[containerIndex].cardinality = 0;
        containers[containerIndex].type = ContainerType::Bitset;
        containers[containerIndex].data = (uint32_t*)malloc(sizeof(uint32_t) * 8192);
        containers[containerIndex].capacity = 8192;

        printf("malloc: containers[%d].data at %p\n", containerIndex, containers[containerIndex].data);
    }
    Container& c = containers[containerIndex];
    switch (c.type)
    {
        case ContainerType::Bitset:
            bitset_setBit(c, offset, value);
            break;

        case ContainerType::Array:
            array_setBit(c, offset, value);
            break;
    }
}

}  // namespace tora::roaring