#pragma once

#include "roaring.cuh"

namespace tora::roaring
{

RoaringBitmapDevice getRandomRoaringBitmap(
    int containerLow, int containerHigh, int numArrays, int numBitsets, int arrayElementLow = 1,
    int arrayElementHigh = 2048);

RoaringBitmapDevice getIntermediateBitmap(int containerLow, int containerHigh);

}  // namespace tora::roaring