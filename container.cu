#include "bitop.cuh"
#include "container.cuh"
#include "memory.cuh"
#include <cassert>

namespace tora::roaring
{

__host__ __device__ Container bitset_bitset_union(const Container& c1, const Container& c2)
{
    Container dst;
    int minLen = c1.capacity < c2.capacity ? c1.capacity : c2.capacity;
    int maxLen = c1.capacity + c2.capacity - minLen;

    dst.data = (uint32_t*)custom_malloc(maxLen * sizeof(uint32_t));
    dst.type = ContainerType::Bitset;
    dst.capacity = maxLen;

    for (int i = 0; i < minLen; i++)
    {
        dst.data[i] = c1.data[i] | c2.data[i];
    }

    const Container* r = c1.capacity > minLen ? &c1 : &c2;
    for (int i = minLen; i < maxLen; i++)
    {
        dst.data[i] = r->data[i];
    }

    return dst;
}

__host__ __device__ Container bitset_bitset_intersect(const Container& c1, const Container& c2)
{
    Container dst;
    int minLen = c1.capacity < c2.capacity ? c1.capacity : c2.capacity;

    dst.data = (uint32_t*)custom_malloc(minLen * sizeof(uint32_t));
    dst.type = ContainerType::Bitset;
    dst.capacity = minLen;

    for (int i = 0; i < minLen; i++)
    {
        dst.data[i] = c1.data[i] & c2.data[i];
    }

    // TODO: convert it to array container if cardinality < 4K.

    return dst;
}

__host__ __device__ Container array_bitset_union(const Container& c1, const Container& c2)
{
    const Container& arr = c1.type == ContainerType::Array ? c1 : c2;
    const Container& bitset = c1.type == ContainerType::Bitset ? c1 : c2;

    Container dst;
    uint16_t* arrayElements = (uint16_t*)arr.data;
    int requiredCapacity = bitset.capacity;
    if (requiredCapacity * sizeof(uint32_t) < arrayElements[arr.cardinality - 1])
    {
        requiredCapacity = (arrayElements[arr.cardinality - 1] + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    }

    dst.data = (uint32_t*)custom_malloc(requiredCapacity * sizeof(uint32_t));

    for (int i = 0; i < bitset.capacity; i++)
    {
        dst.data[i] = bitset.data[i];
    }

    dst.type = ContainerType::Bitset;
    dst.capacity = requiredCapacity;

    int j = 0;
    for (int i = 0; i < arr.cardinality; i++)
    {
        uint16_t element = arrayElements[i];
        int offset = element >> 5;
        int bitpos = element & 31;
        dst.data[offset] |= 1 << bitpos;
        j += 1 - ((bitset.data[offset] & (1 << bitpos)) > 0);
    }

    dst.cardinality = bitset.cardinality + j;

    return dst;
}

__host__ __device__ Container array_bitset_intersect(const Container& c1, const Container& c2)
{
    const Container& arr = c1.type == ContainerType::Array ? c1 : c2;
    const Container& bitset = c1.type == ContainerType::Bitset ? c1 : c2;

    Container dst;
    uint16_t* arrayElements = (uint16_t*)arr.data;
    int requiredCapacity = arr.capacity;

    dst.data = (uint32_t*)custom_malloc(requiredCapacity * sizeof(uint32_t));
    dst.type = ContainerType::Array;
    dst.capacity = requiredCapacity;
    uint16_t* dstElements = (uint16_t*)dst.data;

    int j = 0;
    for (int i = 0; i < arr.cardinality; i++)
    {
        uint16_t element = arrayElements[i];
        int offset = element >> 5;
        int bitpos = element & 31;
        if (bitset.data[offset] & (1 << bitpos))
        {
            dstElements[j++] = element;
        }
    }

    dst.cardinality = j;

    return dst;
}

__host__ __device__ Container array_array_union(const Container& c1, const Container& c2)
{
    Container dst;
    uint16_t* a1 = (uint16_t*)c1.data;
    uint16_t* a2 = (uint16_t*)c2.data;
    int requiredCapacity = (c1.cardinality + c2.cardinality) * 2;  // `sizeof(uint32_t) / sizeof(uint16_t)`

    dst.data = (uint32_t*)custom_malloc(requiredCapacity * sizeof(uint32_t));
    dst.type = ContainerType::Array;
    dst.capacity = requiredCapacity;

    uint16_t* dstElements = (uint16_t*)dst.data;
    int i = 0, j = 0, k = 0;
    while (i < c1.cardinality && j < c2.cardinality)
    {
        if (a1[i] == a2[j])
        {
            dstElements[k++] = a1[i];
            i++;
            j++;
        }
        else if (a1[i] < a2[j])
        {
            dstElements[k++] = a1[i++];
        }
        else
        {
            dstElements[k++] = a2[j++];
        }
    }

    while (i < c1.cardinality)
    {
        dstElements[k++] = a1[i++];
    }

    while (j < c2.cardinality)
    {
        dstElements[k++] = a2[j++];
    }

    dst.cardinality = k;

    // TODO: convert to bitset when cardinality is high

    return dst;
}

__host__ __device__ Container array_array_intersect(const Container& c1, const Container& c2)
{
    Container dst;
    uint16_t* a1 = (uint16_t*)c1.data;
    uint16_t* a2 = (uint16_t*)c2.data;
    int requiredCapacity = (c1.cardinality < c2.cardinality ? c1.cardinality : c2.cardinality) *
                           2;  // `sizeof(uint32_t) / sizeof(uint16_t)`

    dst.data = (uint32_t*)custom_malloc(requiredCapacity * sizeof(uint32_t));
    dst.type = ContainerType::Array;
    dst.capacity = requiredCapacity;

    uint16_t* dstElements = (uint16_t*)dst.data;
    int i = 0, j = 0, k = 0;
    while (i < c1.cardinality && j < c2.cardinality)
    {
        if (a1[i] == a2[j])
        {
            dstElements[k++] = a1[i];
            i++;
            j++;
        }
        else if (a1[i] < a2[j])
        {
            a1[i++];
        }
        else
        {
            a2[j++];
        }
    }

    dst.cardinality = k;

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

__host__ __device__ void bitset_setBit(const Container& c, int offset, bool value)
{
    int index = offset >> 5;
    assert(c.capacity > index);
    c.data[index] |= (1 << (offset & 31));
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

__host__ __device__ static int binarySearch(uint16_t* arr, uint32_t size, uint16_t value) {
    int low = 0;
    int high = size - 1;
    int mid;

    while (low <= high) {
        mid = low + (high - low) / 2;
        if (arr[mid] == value) {
            return mid; // Return the index of the element if found
        } else if (arr[mid] < value) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1; // Return -1 if the element is not found
}

__host__ __device__ static void removeElement(uint16_t* arr, uint32_t& size, uint16_t value) {
    int index = binarySearch(arr, size, value);
    if (index == -1) return; // If the element does not exist, do nothing

    for (int i = index; i < size - 1; ++i) {
        arr[i] = arr[i + 1];
    }
    --size;
}

__host__ __device__ bool array_getBit(const Container& c, int offset)
{
    uint16_t* arr = (uint16_t*)c.data;
    int pos = offset & 65535;
    int index = binarySearch(arr, c.cardinality, pos);
    return index != -1;
}

__host__ __device__ void array_setBit(Container& c, int offset, bool value)
{
    uint16_t pos = offset & 65535;
    uint16_t* arr = (uint16_t*)c.data;
    if (value)
    {
        insertUnique(arr, c.cardinality, c.capacity * 2, pos);
    }
    else
    {
        removeElement(arr, c.cardinality, pos);
    }
}

}  // namespace tora::roaring