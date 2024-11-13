#include "roaring.cuh"

namespace tora::roaring
{

static const int BitmapFlatSize = 65536;

RoaringBitmapDevice::RoaringBitmapDevice()
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;
    deviceData_ = createBitmapFlatOnDevice();
    initBitmapContainers<<<blocksPerGrid, threadsPerBlock>>>();
}

RoaringBitmapDevice::RoaringBitmapDevice(int stream)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (BitmapFlatSize + threadsPerBlock - 1) / threadsPerBlock;
    deviceData_ = createBitmapFlatOnDevice();
    initBitmapContainers<<<blocksPerGrid, threadsPerBlock, stream>>>();
}

RoaringBitmapDevice::~RoaringBitmapDevice()
{
    freeBitmapContainers<<<blocksPerGrid, threadsPerBlock>>>();
    cudaFree(deviceData_);
    deviceData_ = nullptr;
}

__host__ RoaringBitmapFlat* createBitmapFlatOnDevice()
{
    RoaringBitmapFlat* p;
    cudaMalloc((void**)&p, sizeof(RoaringBitmapFlat));
    return p;
}

__global__ void initBitmapContainers(RoaringBitmapFlat* bitmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        bitmap.container[idx] = nullptr;
    }
}

__global__ void freeBitmapContainers(RoaringBitmapFlat* bitmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        if (bitmap.container[idx] != nullptr)
        {
            free(bitmap.container[idx]);
            bitmap.container[idx] = nullptr;
        }
    }
}

__device__ inline int typePair(ContainerType a, ContainerType b)
{
    return (1 << (int)a) + (1 << (int)b);
}

__global__ void bitmapUnion(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, const RoaringBitmapFlat& t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        if (a.containers[idx] != nullptr && b.containers[idx] != nullptr)
        {
            const Container& c1 = a.container[idx];
            const Container& c2 = b.container[idx];
            switch (typePair(c1.type, c2.type))
            {
                case typePair(ContainerType::Bitset, ContainerType::Bitset):
                {
                    t.container[idx] = bitset_bitset_union(c1, c2);
                } break;

                case typePair(ContainerType::Bitset, ContainerType::Array):
                {
                    t.container[idx] = array_bitset_union(c1, c2);
                } break;

                case typePair(ContainerType::Array, ContainerType::Array):
                {
                    t.container[idx] = array_array_union(c1, c2);
                } break;
            }
        }
    }
}

__global__ void bitmapIntersect(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, const RoaringBitmapFlat& t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BitmapFlatSize)
    {
        if (a.containers[idx] != nullptr && b.containers[idx] != nullptr)
        {
            const Container& c1 = a.container[idx];
            const Container& c2 = b.container[idx];
            switch (typePair(c1.type, c2.type))
            {
                case typePair(ContainerType::Bitset, ContainerType::Bitset):
                {
                    t.container[idx] = bitset_bitset_intersect(c1, c2);
                } break;

                case typePair(ContainerType::Bitset, ContainerType::Array):
                {
                    t.container[idx] = array_bitset_intersect(c1, c2);
                } break;

                case typePair(ContainerType::Array, ContainerType::Array):
                {
                    t.container[idx] = array_array_intersect(c1, c2);
                } break;
            }
        }
    }
}

}  // namespace tora::roaring