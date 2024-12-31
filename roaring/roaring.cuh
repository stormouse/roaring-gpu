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
    inline RoaringBitmapFlat& dev() { return *deviceData_; }
    bool getBit(uint32_t pos);
    void setBit(uint32_t pos, bool value);

private:
    RoaringBitmapFlat* deviceData_;
    void free();
};

__global__ void bitmapUnion(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t);
__global__ void bitmapIntersect(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t);
__global__ void bitmapUnionNoAlloc(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t, int containerLow, int containerHigh);
__global__ void bitmapIntersectNoAlloc(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, RoaringBitmapFlat& t, int containerLow, int containerHigh);
__global__ void bitmapGetCardinality(const RoaringBitmapFlat& bitmap, uint32_t* outValue, int containerLow, int containerHigh);
__global__ void bitmapGetBit(const RoaringBitmapFlat& bitmap, uint32_t pos, bool* outValue);
__global__ void bitmapSetBit(RoaringBitmapFlat& bitmap, uint32_t pos, bool value);

}  // namespace tora::roaring