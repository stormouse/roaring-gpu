#pragma once

#include <cstdint>
#include <vector>
#include "container.cuh"

namespace tora::roaring
{

struct RoaringBitmapFlat
{
    Container* containers;  // size = 65536
    __host__ __device__ bool getBit(uint32_t pos) const;
    __host__ __device__ void setBit(uint32_t pos, bool value);
};

class RoaringBitmapDevice
{
public:
    RoaringBitmapDevice();
    explicit RoaringBitmapDevice(int stream);
    ~RoaringBitmapDevice();
    
    RoaringBitmapDevice(const RoaringBitmapDevice&) = delete;

    RoaringBitmapDevice(RoaringBitmapDevice&& other) : deviceData_{other.deviceData_} {
        other.deviceData_ = nullptr;
    }

    RoaringBitmapDevice& operator=(RoaringBitmapDevice&& other){
        this->free();
        cudaDeviceSynchronize();
        this->deviceData_ = other.deviceData_;
        other.deviceData_ = nullptr;
        return *this;
    }

    inline RoaringBitmapFlat* devPtr() { return deviceData_; }
    bool getBit(uint32_t pos);
    void setBit(uint32_t pos, bool value);

private:
    RoaringBitmapFlat* deviceData_;
    void free();
};

__global__ void allocateFlatContainers(RoaringBitmapFlat& a);
__global__ void initBitmapContainers(RoaringBitmapFlat* bitmap);
__global__ void freeBitmapContainers(RoaringBitmapFlat* bitmap);
__global__ void freeFlatContainers(RoaringBitmapFlat* a);

RoaringBitmapFlat* createBitmap();

__global__ void bitmapUnion(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t);
__global__ void bitmapIntersect(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t);
__global__ void bitmapGetBit(const RoaringBitmapFlat& a, uint32_t pos, bool* outValue);
__global__ void bitmapSetBit(RoaringBitmapFlat& a, uint32_t pos, bool value);

}  // namespace tora::roaring