#include <gtest/gtest.h>
#include <bitop.cuh>
#include <container.cuh>
#include <roaring.cuh>

namespace tora::roaring
{

class ContainerTest : public testing::Test
{
protected:
    void TearDown() override
    {
        if (c1.data != nullptr)
        {
            free(c1.data);
        }

        if (c2.data != nullptr)
        {
            free(c2.data);
        }

        if (result.data != nullptr)
        {
            free(result.data);
        }
    }

public:
    Container c1;
    Container c2;
    Container result;
};

TEST_F(ContainerTest, BitsetBitsetUnion)
{
    c1.type = ContainerType::Bitset;
    c2.type = ContainerType::Bitset;

    c1.data = (uint32_t*)malloc(4 * sizeof(uint32_t));
    c2.data = (uint32_t*)malloc(4 * sizeof(uint32_t));

    c1.data[0] = 0xF0F0F0F0;
    c2.data[0] = 0x11111111;

    c1.data[1] = 0xFFFFFFFF;
    c2.data[1] = 0x11111111;

    c1.data[2] = 0xFFFFFFFF;
    c2.data[2] = 0x11111111;

    c1.data[3] = 0xFFFFFFFF;
    c2.data[3] = 0x11111110;

    c1.capacity = 4;
    c1.cardinality = 16 + 32 * 3;

    c2.capacity = 4;
    c2.cardinality = 31;

    result = bitset_bitset_union(c1, c2);
    EXPECT_EQ(20, bitsSet(result.data[0]));
    EXPECT_EQ(32, bitsSet(result.data[1]));
    EXPECT_EQ(32, bitsSet(result.data[2]));
    EXPECT_EQ(32, bitsSet(result.data[3]));
}

TEST_F(ContainerTest, BitsetBitsetIntersect)
{
    c1.type = ContainerType::Bitset;
    c2.type = ContainerType::Bitset;

    c1.data = (uint32_t*)malloc(4 * sizeof(uint32_t));
    c2.data = (uint32_t*)malloc(4 * sizeof(uint32_t));

    c1.data[0] = 0xFFFFFFFF;
    c2.data[0] = 0x11111111;

    c1.data[1] = 0xFFFFFFFF;
    c2.data[1] = 0x11111111;

    c1.data[2] = 0xFFFFFFFF;
    c2.data[2] = 0x11111111;

    c1.data[3] = 0xFFFFFFFF;
    c2.data[3] = 0x11111110;

    c1.capacity = 4;
    c1.cardinality = 32 * 4;

    c2.capacity = 4;
    c2.cardinality = 31;

    result = bitset_bitset_intersect(c1, c2);
    EXPECT_EQ(8, bitsSet(result.data[0]));
    EXPECT_EQ(8, bitsSet(result.data[1]));
    EXPECT_EQ(8, bitsSet(result.data[2]));
    EXPECT_EQ(7, bitsSet(result.data[3]));
}

TEST_F(ContainerTest, ArrayBitsetUnion)
{
    c1.type = ContainerType::Array;
    c2.type = ContainerType::Bitset;

    c1.data = (uint32_t*)malloc(8 * sizeof(uint32_t));
    c2.data = (uint32_t*)malloc(4 * sizeof(uint32_t));

    uint16_t* a1 = (uint16_t*)c1.data;

    for (int i = 0; i < 15; i++)
    {
        a1[i] = 27 + i;
    }

    c2.data[0] = 0x11111111;
    c2.data[1] = 0x11111111;
    c2.data[2] = 0x11111111;
    c2.data[3] = 0x11111111;

    c1.capacity = 8;
    c1.cardinality = 15;

    c2.capacity = 4;
    c2.cardinality = 32;

    result = array_bitset_union(c1, c2);
    EXPECT_EQ(4178645265UL, result.data[0]);
    EXPECT_EQ(286331903UL, result.data[1]);
    EXPECT_EQ(286331153UL, result.data[2]);
    EXPECT_EQ(286331153UL, result.data[3]);
    EXPECT_EQ(12, bitsSet(result.data[0]));
    EXPECT_EQ(15, bitsSet(result.data[1]));
    EXPECT_EQ(8, bitsSet(result.data[2]));
    EXPECT_EQ(8, bitsSet(result.data[3]));
}

TEST_F(ContainerTest, ArrayBitsetIntersect)
{
    c1.type = ContainerType::Array;
    c2.type = ContainerType::Bitset;

    c1.data = (uint32_t*)malloc(8 * sizeof(uint32_t));
    c2.data = (uint32_t*)malloc(4 * sizeof(uint32_t));

    uint16_t* a1 = (uint16_t*)c1.data;

    for (int i = 0; i < 15; i++)
    {
        a1[i] = 27 + i;
    }

    c2.data[0] = 0x11111111;
    c2.data[1] = 0x11111111;
    c2.data[2] = 0x11111111;
    c2.data[3] = 0x11111111;

    c1.capacity = 8;
    c1.cardinality = 15;

    c2.capacity = 4;
    c2.cardinality = 32;

    result = array_bitset_intersect(c1, c2);
    uint16_t* arrayElements = (uint16_t*)result.data;

    EXPECT_EQ(4, result.cardinality);
    EXPECT_EQ(28, arrayElements[0]);
    EXPECT_EQ(32, arrayElements[1]);
    EXPECT_EQ(36, arrayElements[2]);
    EXPECT_EQ(40, arrayElements[3]);
}

TEST_F(ContainerTest, ArrayArrayUnion)
{
    c1.type = ContainerType::Array;
    c2.type = ContainerType::Array;

    c1.data = (uint32_t*)malloc(20 * sizeof(uint32_t));
    c2.data = (uint32_t*)malloc(20 * sizeof(uint32_t));

    uint16_t* a1 = (uint16_t*)c1.data;
    uint16_t* a2 = (uint16_t*)c2.data;

    for (int i = 0; i < 40; i++)
    {
        a1[i] = 9 + 7 * i;  // 9, 16, ..., 275, 282
        a2[i] = 8 + 4 * i;  // 8, 12, ..., 160, 164
        // overlaps: 16, 44, 72, 100, 128, 156 (6 overlaps)
    }

    c1.capacity = 20;
    c1.cardinality = 40;

    c2.capacity = 20;
    c2.cardinality = 40;

    result = array_array_union(c1, c2);
    uint16_t* arrayElements = (uint16_t*)result.data;
    EXPECT_EQ(74, result.cardinality);
    int idx = 0;
    for (int i = 8; i <= 164; i++)
    {
        if (((i - 9) % 7 == 0) || ((i - 8) % 4 == 0))
        {
            EXPECT_EQ(i, arrayElements[idx]);
            idx++;
        }
    }
}

TEST_F(ContainerTest, ArrayArrayIntersect)
{
    c1.type = ContainerType::Array;
    c2.type = ContainerType::Array;

    c1.data = (uint32_t*)malloc(20 * sizeof(uint32_t));
    c2.data = (uint32_t*)malloc(20 * sizeof(uint32_t));

    uint16_t* a1 = (uint16_t*)c1.data;
    uint16_t* a2 = (uint16_t*)c2.data;

    for (int i = 0; i < 40; i++)
    {
        a1[i] = 9 + 7 * i;  // 9, 16, ..., 275, 282
        a2[i] = 8 + 4 * i;  // 8, 12, ..., 160, 164
        // overlaps: 16, 44, 72, 100, 128, 156 (6 overlaps)
    }

    c1.capacity = 20;
    c1.cardinality = 40;

    c2.capacity = 20;
    c2.cardinality = 40;

    result = array_array_intersect(c1, c2);
    uint16_t* arrayElements = (uint16_t*)result.data;

    EXPECT_EQ(6, result.cardinality);
    EXPECT_EQ(16, arrayElements[0]);
    EXPECT_EQ(44, arrayElements[1]);
    EXPECT_EQ(72, arrayElements[2]);
    EXPECT_EQ(100, arrayElements[3]);
    EXPECT_EQ(128, arrayElements[4]);
    EXPECT_EQ(156, arrayElements[5]);
}
};  // namespace tora::roaring