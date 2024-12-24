#include <stdio.h>
#include "cuda_common.cuh"
#include "memory.cuh"
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

void RoaringBitmapDevice::free()
{
    if (deviceData_ != nullptr)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;
        freeBitmapContainers<<<blocksPerGrid, threadsPerBlock>>>(deviceData_);
        freeFlatContainers<<<1, 1>>>(deviceData_);
        checkCuda(cudaFree(deviceData_));
    }
}

RoaringBitmapDevice::~RoaringBitmapDevice()
{
    free();
}

__global__ void allocateFlatContainers(RoaringBitmapFlat& a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        a.containers = (Container*)custom_malloc(65536 * sizeof(Container));
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
            custom_free(bitmap->containers[idx].data);
        }
    }
}

__global__ void freeFlatContainers(RoaringBitmapFlat* a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        custom_free(a->containers);
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
            }
            break;

            case typePair(ContainerType::Bitset, ContainerType::Array):
            {
                t.containers[idx] = array_bitset_union(c1, c2);
            }
            break;

            case typePair(ContainerType::Array, ContainerType::Array):
            {
                t.containers[idx] = array_array_union(c1, c2);
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

__global__ void bitmapGetBit(const RoaringBitmapFlat& a, uint32_t pos, bool* outValue)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        *outValue = a.getBit(pos);
    }
}

__global__ void bitmapSetBit(RoaringBitmapFlat& a, uint32_t pos, bool value)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        a.setBit(pos, value);
    }
}

__host__ __device__ bool RoaringBitmapFlat::getBit(uint32_t pos) const
{
    int containerIndex = (pos >> 16);
    int offset = pos & 0xFFFF;
    if (containers[containerIndex].data == nullptr || containers[containerIndex].cardinality == 0)
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

__host__ __device__ void RoaringBitmapFlat::setBit(uint32_t pos, bool value)
{
    int containerIndex = (pos >> 16);
    int offset = pos & 0xFFFF;
    if (containers[containerIndex].data == nullptr)
    {
        containers[containerIndex].cardinality = 0;
        containers[containerIndex].type = ContainerType::Array;
        containers[containerIndex].data = (uint32_t*)custom_malloc(sizeof(uint32_t) * 2048);
        containers[containerIndex].capacity = 2048;
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

bool RoaringBitmapDevice::getBit(uint32_t pos)
{
    bool outPut;
    bool* outputDevice;
    checkCuda(cudaMallocHost((void**)&outputDevice, sizeof(bool)));
    bitmapGetBit<<<1, 1>>>(*deviceData_, pos, outputDevice);
    checkCuda(cudaMemcpy(&outPut, outputDevice, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return outPut;
}

void RoaringBitmapDevice::setBit(uint32_t pos, bool value)
{
    bitmapSetBit<<<1, 1>>>(*deviceData_, pos, value);
}

}  // namespace tora::roaring