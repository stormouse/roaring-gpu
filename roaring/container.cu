#include <stdio.h>
#include <cassert>
#include "bitop.cuh"
#include "container.cuh"
#include "memory.cuh"

namespace tora::roaring
{

__host__ __device__ void Container::clear()
{
    cardinality = 0;
}

__host__ __device__ void Container::zero()
{
    cardinality = 0;
    memset(data, 0, sizeof(uint32_t) * capacity);
}

__host__ __device__ Container bitset_bitset_union(const Container& c1, const Container& c2)
{
    Container dst;
    dst.data = (uint32_t*)custom_malloc(65536);
    dst.capacity = 65536 / static_cast<int>(sizeof(uint32_t));
    bitset_bitset_union(c1, c2, dst);
    return dst;
}

__host__ __device__ Container bitset_bitset_intersect(const Container& c1, const Container& c2)
{
    Container dst;
    dst.data = (uint32_t*)custom_malloc(65536);
    dst.capacity = 65536 / static_cast<int>(sizeof(uint32_t));
    bitset_bitset_intersect(c1, c2, dst);
    return dst;
}

__host__ __device__ Container array_bitset_union(const Container& c1, const Container& c2)
{
    Container dst;
    dst.data = (uint32_t*)custom_malloc(65536);
    dst.capacity = 65536 / static_cast<int>(sizeof(uint32_t));
    array_bitset_union(c1, c2, dst);

    return dst;
}

__host__ __device__ Container array_bitset_intersect(const Container& c1, const Container& c2)
{
    Container dst;
    dst.data = (uint32_t*)custom_malloc(65536);
    dst.capacity = 65536 / static_cast<int>(sizeof(uint32_t));
    array_bitset_intersect(c1, c2, dst);
    return dst;
}

__host__ __device__ Container array_array_union(const Container& c1, const Container& c2)
{
    Container dst;
    dst.data = (uint32_t*)custom_malloc(65536);
    dst.capacity = 65536 / static_cast<int>(sizeof(uint32_t));
    array_array_union(c1, c2, dst);
    return dst;
}

__host__ __device__ Container array_array_intersect(const Container& c1, const Container& c2)
{
    Container dst;
    dst.data = (uint32_t*)custom_malloc(65536);
    dst.capacity = 65536 / static_cast<int>(sizeof(uint32_t));
    array_array_intersect(c1, c2, dst);
    return dst;
}

// __device__ cannot be directly applied to constructors
__host__ __device__ Container cloneContainer(const Container& original)
{
    Container clone;
    clone.type = original.type;
    clone.cardinality = original.cardinality;
    clone.capacity = original.capacity;
    clone.data = (uint32_t*)custom_malloc(clone.capacity * sizeof(uint32_t));
    for (int i = 0; i < clone.capacity; i++)
    {
        clone.data[i] = original.data[i];
    }
    return clone;
}

__host__ __device__ bool bitset_getBit(const Container& c, int offset)
{
    int index = offset >> 5;
    if (c.capacity <= index)
        return false;
    return (c.data[index] & (1 << (offset & 31))) != 0;
}

__host__ __device__ void bitset_setBit(Container& c, int offset, bool value)
{
    int index = offset >> 5;
    int bit = 1 << (offset & 0x1F);
    assert(c.capacity > index);

    if ((c.data[index] & bit) == 0)
    {
        c.data[index] |= bit;
        c.cardinality++;
    }
}

__host__ __device__ static int findInsertPosition(uint16_t* arr, uint32_t size, uint16_t value)
{
    int low = 0;
    int high = size - 1;
    int mid;

    while (low <= high)
    {
        mid = low + (high - low) / 2;
        if (arr[mid] == value)
        {
            return -1;  // Value already exists, return -1 to indicate no insertion needed
        }
        else if (arr[mid] < value)
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
    return low;
}

__host__ __device__ static void insertUnique(uint16_t arr[], uint32_t& size, int capacity, uint16_t value)
{
    if (size == 0)
    {
        arr[size++] = value;
        return;
    }

    int position = findInsertPosition(arr, size, value);
    if (position == -1)
        return;  // If the value already exists, do not insert

    if (size < capacity)
    {
        for (int i = size; i > position; --i)
        {
            arr[i] = arr[i - 1];
        }
        arr[position] = value;
        ++size;
    }
}

__host__ __device__ static int binarySearch(uint16_t* arr, uint32_t size, uint16_t value)
{
    int low = 0;
    int high = size - 1;
    int mid;

    while (low <= high)
    {
        mid = low + (high - low) / 2;
        if (arr[mid] == value)
        {
            return mid;  // Return the index of the element if found
        }
        else if (arr[mid] < value)
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
    return -1;  // Return -1 if the element is not found
}

__host__ __device__ static void removeElement(uint16_t* arr, uint32_t& size, uint16_t value)
{
    int index = binarySearch(arr, size, value);
    if (index == -1)
        return;  // If the element does not exist, do nothing

    for (int i = index; i < size - 1; ++i)
    {
        arr[i] = arr[i + 1];
    }
    --size;
}

__host__ __device__ bool array_getBit(const Container& c, int offset)
{
    uint16_t* arr = (uint16_t*)c.data;
    int index = binarySearch(arr, c.cardinality, offset);
    return index != -1;
}

__host__ __device__ void array_setBit(Container& c, int offset, bool value)
{
    uint16_t* arr = (uint16_t*)c.data;
    if (value)
    {
        insertUnique(arr, c.cardinality, c.capacity * 2, offset);
    }
    else
    {
        removeElement(arr, c.cardinality, offset);
    }
}

__host__ __device__ void bitset_bitset_union(const Container& a, const Container& b, Container& dst)
{
    dst.clear();

    if (a.cardinality == 0 && b.cardinality == 0)
    {
        return;
    }

    int minLen = min(a.capacity, b.capacity);
    int maxLen = a.capacity + b.capacity - minLen;
    assert(dst.capacity >= maxLen);

    dst.type = ContainerType::Bitset;
    dst.cardinality = 0;

    for (int i = 0; i < minLen; i++)
    {
        dst.data[i] = a.data[i] | b.data[i];
        dst.cardinality += bitsSet(dst.data[i]);
    }

    const Container* r = a.capacity > minLen ? &a : &b;
    for (int i = minLen; i < maxLen; i++)
    {
        dst.data[i] = r->data[i];
        dst.cardinality += bitsSet(dst.data[i]);
    }
}

__host__ __device__ void bitset_bitset_intersect(const Container& a, const Container& b, Container& dst)
{
    dst.clear();

    if (a.cardinality == 0 || b.cardinality == 0)
    {
        return;
    }

    int len = min(a.capacity, b.capacity);
    assert(dst.capacity >= len);

    dst.type = ContainerType::Bitset;
    dst.cardinality = 0;

    for (int i = 0; i < len; i++)
    {
        dst.data[i] = a.data[i] & b.data[i];
        dst.cardinality += bitsSet(dst.data[i]);
    }

    // TODO: convert to array if too few bits are set
}

__host__ __device__ void array_bitset_union(const Container& a, const Container& b, Container& dst)
{
    dst.clear();

    if (a.cardinality == 0 && b.cardinality == 0)
    {
        return;
    }

    const Container& arr = a.type == ContainerType::Array ? a : b;
    const Container& bitset = a.type == ContainerType::Bitset ? a : b;
    uint16_t* arrayElements = (uint16_t*)arr.data;
    assert(&arr != &bitset);
    assert(dst.capacity >= bitset.capacity);

    dst.type = ContainerType::Bitset;
    memcpy(dst.data, bitset.data, bitset.capacity * sizeof(uint32_t));

    int flips = 0;
    for (int i = 0; i < arr.cardinality; i++)
    {
        uint16_t element = arrayElements[i];
        int offset = element >> 5;
        int bitpos = element & 0x1F;
        dst.data[offset] |= 1 << bitpos;
        flips += 1 - ((bitset.data[offset] & (1 << bitpos)) > 0);
    }
    dst.cardinality = bitset.cardinality + flips;
}

__host__ __device__ void array_bitset_intersect(const Container& a, const Container& b, Container& dst)
{
    dst.clear();

    if (a.cardinality == 0 || b.cardinality == 0)
    {
        return;
    }

    const Container& arr = a.type == ContainerType::Array ? a : b;
    const Container& bitset = a.type == ContainerType::Bitset ? a : b;
    assert(&arr != &bitset);

    dst.type = ContainerType::Array;
    uint16_t* arrayElements = (uint16_t*)arr.data;
    uint16_t* dstElements = (uint16_t*)dst.data;

    int inc = 0;
    for (int i = 0; i < arr.cardinality; i++)
    {
        uint16_t element = arrayElements[i];
        int offset = element >> 5;
        int bitpos = element & 0x1F;
        if (bitset.data[offset] & (1 << bitpos))
        {
            dstElements[inc++] = element;
        }
    }
    dst.cardinality = inc;
}

__host__ __device__ void array_array_union_bitset(const Container& a, const Container& b, Container& dst)
{
    dst.zero();
    dst.type = ContainerType::Bitset;

    uint16_t* aa = (uint16_t*)a.data;
    uint16_t* bb = (uint16_t*)b.data;

    int inc = 0;
    for (int i = 0; i < a.cardinality; i++)
    {
        uint16_t element = aa[i];
        int offset = element >> 5;
        int bitpos = element & 0x1F;
        inc += 1 - ((dst.data[offset] & (1 << bitpos)) > 0);
        dst.data[offset] |= 1 << bitpos;
    }

    for (int i = 0; i < b.cardinality; i++)
    {
        uint16_t element = bb[i];
        int offset = element >> 5;
        int bitpos = element & 0x1F;
        inc += 1 - ((dst.data[offset] & (1 << bitpos)) > 0);
        dst.data[offset] |= 1 << bitpos;
    }

    dst.cardinality = inc;
}

__host__ __device__ void array_array_union(const Container& a, const Container& b, Container& dst)
{
    dst.clear();

    if (a.cardinality == 0 && b.cardinality == 0)
    {
        return;
    }

    uint16_t* aa = (uint16_t*)a.data;
    uint16_t* bb = (uint16_t*)b.data;

    // if exceeds 65536 bits we will convert it to bitset container
    int requiredCapacity = min(65536 / static_cast<int>(sizeof(uint32_t)), (a.cardinality + b.cardinality) * 2);
    assert(dst.capacity >= requiredCapacity);

    dst.type = ContainerType::Array;

    uint16_t* dstElements = (uint16_t*)dst.data;
    int i = 0, j = 0, k = 0;
    while (i < a.cardinality && j < b.cardinality)
    {
        if (aa[i] == bb[j])
        {
            dstElements[k++] = aa[i];
            i++;
            j++;
        }
        else if (aa[i] < bb[j])
        {
            dstElements[k++] = aa[i++];
        }
        else
        {
            dstElements[k++] = bb[j++];
        }

        if (k == dst.capacity * 2)
        {
            array_array_union_bitset(a, b, dst);
            return;
        }
    }

    while (i < a.cardinality)
    {
        dstElements[k++] = aa[i++];
        if (k == dst.capacity * 2)
        {
            array_array_union_bitset(a, b, dst);
            return;
        }
    }

    while (j < b.cardinality)
    {
        dstElements[k++] = bb[j++];
        if (k == dst.capacity * 2)
        {
            array_array_union_bitset(a, b, dst);
            return;
        }
    }

    dst.cardinality = k;
}

__host__ __device__ void array_array_intersect(const Container& a, const Container& b, Container& dst)
{
    dst.clear();

    if (a.cardinality == 0 || b.cardinality == 0)
    {
        return;
    }

    uint16_t* aa = (uint16_t*)a.data;
    uint16_t* bb = (uint16_t*)b.data;
    int requiredCapacity = min(a.cardinality + 1, b.cardinality + 1) >> 1;
    assert(dst.capacity >= requiredCapacity);

    dst.type = ContainerType::Array;
    uint16_t* dstElements = (uint16_t*)dst.data;
    int i = 0, j = 0, k = 0;
    while (i < a.cardinality && j < b.cardinality)
    {
        if (aa[i] == bb[j])
        {
            dstElements[k++] = aa[i];
            i++;
            j++;
        }
        else if (aa[i] < bb[j])
        {
            i++;
        }
        else
        {
            j++;
        }
    }
    dst.cardinality = k;
}

}  // namespace tora::roaring