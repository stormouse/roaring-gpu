#pragma once

#include <cstdint>
#include <vector>
#include "container.cuh"

namespace tora::roaring
{

struct RoaringBitmapFlat
{
    Container* containers;  // size = 65536
    __host__ __device__ bool getBit(int pos) const;
    __host__ __device__ void setBit(int pos, bool value);
};

class RoaringBitmapDevice
{
public:
    RoaringBitmapDevice();
    explicit RoaringBitmapDevice(int stream);
    ~RoaringBitmapDevice();
    RoaringBitmapDevice(const RoaringBitmapDevice&) = default;
    RoaringBitmapDevice(RoaringBitmapDevice&&) = default;

    inline RoaringBitmapFlat* devPtr() { return deviceData_; }
    bool getBit(int pos);
    void setBit(int pos, bool value);

private:
    RoaringBitmapFlat* deviceData_;
};

__global__ void allocateFlatContainers(RoaringBitmapFlat& a);
__global__ void initBitmapContainers(RoaringBitmapFlat* bitmap);
__global__ void freeBitmapContainers(RoaringBitmapFlat* bitmap);
__global__ void freeFlatContainers(RoaringBitmapFlat* a);

RoaringBitmapFlat* createBitmap();

__global__ void bitmapUnion(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t);
__global__ void bitmapIntersect(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t);
__global__ void bitmapGetBit(const RoaringBitmapFlat& a, int pos, bool* outValue);
__global__ void bitmapSetBit(RoaringBitmapFlat& a, int pos, bool value);

}  // namespace tora::roaring