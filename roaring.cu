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
                }
                break;

                case typePair(ContainerType::Bitset, ContainerType::Array):
                {
                    t.container[idx] = array_bitset_union(c1, c2);
                }
                break;

                case typePair(ContainerType::Array, ContainerType::Array):
                {
                    t.container[idx] = array_array_union(c1, c2);
                }
                break;
            }
        }
        else if (a.container[idx] != nullptr)
        {
            t.container[idx] = cloneContainer(a.container[idx]);
        }
        else if (b.container[idx] != nullptr)
        {
            t.container[idx] = cloneContainer(b.container[idx]);
        }
        else
        {
            t.container[idx] = nullptr;
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
                }
                break;

                case typePair(ContainerType::Bitset, ContainerType::Array):
                {
                    t.container[idx] = array_bitset_intersect(c1, c2);
                }
                break;

                case typePair(ContainerType::Array, ContainerType::Array):
                {
                    t.container[idx] = array_array_intersect(c1, c2);
                }
                break;
            }
        }
        else
        {
            t.container[idx] = nullptr;
        }
    }
}

__global__ void bitmapGetBit(const RoaringBitmapFlat& a, int pos, int* outValue)
{
    if (threadIdx.x == 0)
    {
        int containerIndex = (pos >> 16);
        int offset = pos & 65535;
        if (a.containers[containerIndex] == nullptr)
        {
            outValue = 0;
        }
        else
        {
            const Container& c = a.containers[containerIndex];
            switch (c.type)
            {
                case ContainerType::Bitset:
                    outValue = bitset_getBit(c, offset);
                    break;

                case ContainerType::Array:
                    outValue = array_getBit(c, offset);
                    break;
            }
        }
    }
}
__global__ void bitmapSetBit(RoaringBitmapFlat& a, int pos, bool value)
{
    if (threadIdx.x == 0)
    {
        int containerIndex = (pos >> 16);
        int offset = pos & 65535;
        if (a.containers[containerIndex] == nullptr)
        {
            a.containers[containerIndex] = malloc(sizeof(Container));
            a.containers[containerIndex].cardinality = 1;
            // TODO: save me from my laziness
            a.containers[containerIndex].type = ContainerType::Bitset;
            a.containers[containerIndex].data = malloc(sizeof(uint32_t) * 8192);
            a.containers[containerIndex].capacity = 8192;
        }
        Container& c = a.containers[containerIndex];
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
}

}  // namespace tora::roaring