#include <gtest/gtest.h>
#include <bitop.cuh>
#include <container.cuh>
#include <roaring.cuh>

namespace tora::roaring
{

TEST(BitOpTest, CountBitsSet)
{
    EXPECT_EQ(3, bitsSet(0b01010001));
    EXPECT_EQ(4, bitsSet(0b10010101));
    EXPECT_EQ(4, bitsSet(0b00110110));
    EXPECT_EQ(5, bitsSet(0b10011101));
    EXPECT_EQ(1, bitsSet(0b00100000));
    EXPECT_EQ(0, bitsSet(0));
    EXPECT_EQ(8, bitsSet(0b11111111));
}

TEST(BitOpTest, CountTrailingZeros)
{
    EXPECT_EQ(0, trailingZeros(0b01010001));
    EXPECT_EQ(0, trailingZeros(0b10010101));
    EXPECT_EQ(1, trailingZeros(0b00110110));
    EXPECT_EQ(2, trailingZeros(0b10011100));
    EXPECT_EQ(5, trailingZeros(0b00100000));
    EXPECT_EQ(0, trailingZeros(0b11011111));
    EXPECT_EQ(11, trailingZeros(0b100000000000));
    // EXPECT_EQ(0, trailingZeros(0));
}

};  // namespace tora::roaring