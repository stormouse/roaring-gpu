#include <benchmark/benchmark.h>
#include <random>
#include <roaring_helper.cuh>
#include <string>

using namespace tora::roaring;

static void BM_StringCreation(::benchmark::State& state)
{
    for (auto _ : state)
    {
        std::string empty_string;
    }
}
BENCHMARK(BM_StringCreation);

static void BM_StringCopy(::benchmark::State& state)
{
    std::string x = "hello";
    for (auto _ : state)
    {
        std::string copy(x);
    }
}
BENCHMARK(BM_StringCopy);

static void BitmapCreation(::benchmark::State& state)
{        
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<uint32_t> dist(0U, UINT32_MAX);
    std::vector<int> indexesToQuery;
    for (int i = 0; i < 1000; i++)
    {
        indexesToQuery.push_back(dist(mt));
    }
    auto bitmap = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);

    uint32_t numFlagsSet = 0;
    uint32_t j = 0;
    for (auto _ : state)
    {
        numFlagsSet += bitmap.getBit(j);
        j = (j + 1) % indexesToQuery.size();
    }
}
BENCHMARK(BitmapCreation);

static void BitmapIntersect(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    RoaringBitmapDevice result;

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    for (auto _ : state)
    {
        bitmapIntersect<<<blocksPerGrid, threadsPerBlock>>>(*a.devPtr(), *b.devPtr(), *result.devPtr());
        cudaDeviceSynchronize();
    }
}
BENCHMARK(BitmapIntersect);

static void BitmapUnion(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    RoaringBitmapDevice result;

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    for (auto _ : state)
    {
        bitmapUnion<<<blocksPerGrid, threadsPerBlock>>>(*a.devPtr(), *b.devPtr(), *result.devPtr());
        cudaDeviceSynchronize();
    }
}
BENCHMARK(BitmapUnion);

// BENCHMARK_MAIN();
int main(int argc, char** argv)
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (4096 * 1024 * 1024UL));
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}