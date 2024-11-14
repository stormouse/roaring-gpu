#pragma once

#include <cstdint>
#include <vector>
#include "container.cuh"

namespace tora::roaring
{

// // Implementation of RoaringFormatSpec
// // https://github.com/RoaringBitmap/RoaringFormatSpec/?tab=readme-ov-file#general-layout

// class RoaringBitmap
// {
// public:
//     // construct an empty bitmap
//     RoaringBitmap();

//     // construct a bitmap from serialized bytes
//     explicit RoaringBitmap(void* data);

//     // moving a bitmap is ok
//     RoaringBitmap(RoaringBitmap&& other) noexcept;
//     RoaringBitmap& operator=(RoaringBitmap&& other) noexcept;

//     // copying in memory is forbidden
//     RoaringBitmap(const RoaringBitmap& roaringBitmap) = delete;
//     RoaringBitmap operator=(const RoaringBitmap& roaringBitmap) = delete;
//     RoaringBitmap& operator=(const RoaringBitmap& roaringBitmap) = delete;

//     // get container count
//     __host__ __device__ uint16_t containerCount() const { return containerCount_; }

//     // get list of containers
//     __host__ __device__ const std::vector<Container> containers() const;

//     // get if a bit is set at `pos`
//     __host__ __device__ void getBit(uint32_t pos);

//     // set a bit at `pos` with `value`
//     __host__ __device__ void setBit(uint32_t pos, bool value);

// private:
//     uint16_t containerCount_;
//     uint32_t* descriptiveHeaders_;  // L = containerCount_;
//     uint32_t* offsetHeaders_;       // L = containerCount_;
//     Container* containerStorage_;   // L = containerCount_;
//     Container* denseContainerStorage_; // L = max_value(uint16_t)
// };


struct RoaringBitmapFlat
{
    Container* containers; // size = 65536
    bool getBit(int pos) const;
    void setBit(int pos, bool value);
};

class RoaringBitmapDevice
{
public:
    RoaringBitmapDevice();
    explicit RoaringBitmapDevice(int stream);
    ~RoaringBitmapDevice();

private:
    RoaringBitmapFlat* deviceData_;
};

__host__ RoaringBitmapFlat* createBitmapFlatOnDevice();
__global__ void initBitmapContainers(RoaringBitmapFlat* bitmap);
__global__ void freeBitmapContainers(RoaringBitmapFlat* bitmap);

RoaringBitmapFlat* createBitmap();

__global__ void bitmapUnion(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, const RoaringBitmapFlat& t);
__global__ void bitmapIntersect(const RoaringBitmapFlat& a, const RoaringBitmapFlat& b, const RoaringBitmapFlat& t);
__global__ void bitmapGetBit(const RoaringBitmapFlat& a, int pos, int* outValue);
__global__ void bitmapSetBit(const RoaringBitmapFlat& a, int pos, bool value);

}  // namespace tora::roaring