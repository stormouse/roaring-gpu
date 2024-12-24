#include <gtest/gtest.h>
#include <roaring.cuh>
#include <roaring_helper.cuh>
#include <random>

namespace tora::roaring
{

TEST(RoaringBitmapTest, CreateRandomBitmap)
{
    auto bitmap = getRandomRoaringBitmap(0, 10, 5, 5, 1, 18);

    for (int i = 0; i < 10; i++)
    {
        std::cerr << i << ":";
        for (int j = 0; j < 16; j++)
        {
            std::cerr << bitmap.getBit((i << 16) + j) << ",";
        }
        std::cerr << "\n";
    }

    for (int i = 0; i < 16; i++)
    {
        EXPECT_FALSE(bitmap.getBit((10 << 16) + i));
    }
}

TEST(RoaringBitmapTest, RandomAccessSafety)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<uint32_t> dist(0U, (2048U) << 16);

    std::vector<uint32_t> indexesToQuery;
    for (int i = 0; i < 100; i++)
    {
        indexesToQuery.push_back(dist(mt));
    }
    
    auto bitmap = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);

    int bitsSet = 0;
    for (int i = 0, j = 0; i < 1000; i++, j = (j + 1) % indexesToQuery.size())
    {
        bitsSet += bitmap.getBit(indexesToQuery[j]);
    }

    std::cerr << "Bits set: " << bitsSet << "\n";
}

};  // namespace tora::roaring


int main(int argc, char **argv) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (4096 * 1024 * 1024UL));
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}