#include <gtest/gtest.h>
#include <random>
#include <roaring.cuh>
#include <roaring_helper.cuh>

namespace tora::roaring
{

class BitmapTest : public testing::Test
{
protected:
    void SetUp() override
    {
        b1.containers = new Container[65536];
        b2.containers = new Container[65536];
        b3.containers = new Container[65536];

        for (int i = 0; i < 65535; i++)
        {
            b1.containers[i].type = b2.containers[i].type = b3.containers[i].type = ContainerType::Array;
        }
    }

    void TearDown() override
    {
        if (b1.containers != nullptr)
        {
            for (int i = 0; i < 65535; i++)
            {
                if (b1.containers[i].data != nullptr)
                {
                    free(b1.containers[i].data);
                }
            }
            delete[] b1.containers;
        }

        if (b2.containers != nullptr)
        {
            for (int i = 0; i < 65535; i++)
            {
                if (b2.containers[i].data != nullptr)
                {
                    free(b2.containers[i].data);
                }
            }
            delete[] b2.containers;
        }

        if (b3.containers != nullptr)
        {
            for (int i = 0; i < 65535; i++)
            {
                if (b3.containers[i].data != nullptr)
                {
                    free(b3.containers[i].data);
                }
            }
            delete[] b3.containers;
        }
    }

public:
    RoaringBitmapFlat b1;
    RoaringBitmapFlat b2;
    RoaringBitmapFlat b3;
};

TEST_F(BitmapTest, BitmapGetSet)
{
    for (int i = 0; i < 64; i++)
    {
        b1.setBit(10251 + i, true);
    }

    for (int i = 0; i < 64; i++)
    {
        b2.setBit(10271 + i, true);
    }

    EXPECT_TRUE(b1.getBit(10253));
    EXPECT_TRUE(b1.getBit(10254));
    EXPECT_TRUE(b1.getBit(10255));
    EXPECT_FALSE(b1.getBit(51));
    EXPECT_FALSE(b1.getBit(17959120));
    EXPECT_FALSE(b1.getBit(2179120));
    EXPECT_TRUE(b2.getBit(10271));
    EXPECT_TRUE(b2.getBit(10272));
    EXPECT_TRUE(b2.getBit(10299));
    EXPECT_FALSE(b2.getBit(10254));
    EXPECT_FALSE(b2.getBit(16));
    EXPECT_FALSE(b2.getBit(9));
}

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

int main(int argc, char** argv)
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (4096 * 1024 * 1024UL));
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}