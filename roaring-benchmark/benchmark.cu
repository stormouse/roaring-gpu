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

// static void BitmapCreation(::benchmark::State& state)
// {        
//     std::random_device rd;
//     std::mt19937 mt(rd());
//     std::uniform_int_distribution<uint32_t> dist(0U, UINT32_MAX);
//     std::vector<int> indexesToQuery;
//     for (int i = 0; i < 1000; i++)
//     {
//         indexesToQuery.push_back(dist(mt));
//     }
//     auto bitmap = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);

//     uint32_t numFlagsSet = 0;
//     uint32_t j = 0;
//     for (auto _ : state)
//     {
//         numFlagsSet += bitmap.getBit(j);
//         j = (j + 1) % indexesToQuery.size();
//     }
// }
// BENCHMARK(BitmapCreation);

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

    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(*result.devPtr(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(outValue);
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

    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(*result.devPtr(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(outValue);
}
BENCHMARK(BitmapUnion);


static void BitmapIntersectPreAllocate(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto intermediate = getIntermediateBitmap(0, 2048);

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    for (auto _ : state)
    {
        bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(*a.devPtr(), *b.devPtr(), *intermediate.devPtr(), 0, 2048);
        cudaDeviceSynchronize();
    }

    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(*intermediate.devPtr(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(outValue);
}
BENCHMARK(BitmapIntersectPreAllocate);

static void BitmapUnionPreAllocate(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto intermediate = getIntermediateBitmap(0, 2048);

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    for (auto _ : state)
    {
        bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock>>>(*a.devPtr(), *b.devPtr(), *intermediate.devPtr(), 0, 2048);
        cudaDeviceSynchronize();
    }
    
    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(*intermediate.devPtr(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(outValue);
}
BENCHMARK(BitmapUnionPreAllocate);


static void BitmapIntersectPreAllocateStreamed(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    RoaringBitmapDevice intermediates[] {
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048)
    };

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t streams[4];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);

    int numBatches = state.range(0);

    for (auto _ : state)
    {
        for (int i = 0; i < numBatches; i++)
        {
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[0].dev(), 0, 2048);
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[1]>>>(a.dev(), b.dev(), intermediates[1].dev(), 0, 2048);
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[2]>>>(a.dev(), b.dev(), intermediates[2].dev(), 0, 2048);
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[3]>>>(a.dev(), b.dev(), intermediates[3].dev(), 0, 2048);
        }
        cudaDeviceSynchronize();
    }

    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(intermediates[0].dev(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(outValue);
    
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cudaStreamDestroy(streams[3]);
}
BENCHMARK(BitmapIntersectPreAllocateStreamed)->DenseRange(1, 8, 1);

static void BitmapUnionPreAllocateStreamed(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    RoaringBitmapDevice intermediates[] {
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048)
    };

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t streams[4];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);

    int numBatches = state.range(0);

    for (auto _ : state)
    {
        for (int i = 0; i < numBatches; i++)
        {
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[0].dev(), 0, 2048);
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[1]>>>(a.dev(), b.dev(), intermediates[1].dev(), 0, 2048);
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[2]>>>(a.dev(), b.dev(), intermediates[2].dev(), 0, 2048);
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[3]>>>(a.dev(), b.dev(), intermediates[3].dev(), 0, 2048);
        }
        cudaDeviceSynchronize();
    }
    
    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(intermediates[0].dev(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(outValue);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cudaStreamDestroy(streams[3]);
}
BENCHMARK(BitmapUnionPreAllocateStreamed)->DenseRange(1, 8, 1);



static void BitmapIntersectPreAllocateSingleStream(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    RoaringBitmapDevice intermediates[] {
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048)
    };

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t streams[4];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);

    int numBatches = state.range(0);

    for (auto _ : state)
    {
        for (int i = 0; i < numBatches; i++)
        {
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[0].dev(), 0, 2048);
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[1].dev(), 0, 2048);
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[2].dev(), 0, 2048);
            bitmapIntersectNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[3].dev(), 0, 2048);
        }
        cudaDeviceSynchronize();
    }

    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(intermediates[0].dev(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cudaStreamDestroy(streams[3]);
}
BENCHMARK(BitmapIntersectPreAllocateSingleStream)->DenseRange(1, 3, 1);

static void BitmapUnionPreAllocateSingleStream(::benchmark::State& state)
{
    auto a = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    auto b = getRandomRoaringBitmap(0, 2048, 1024, 1024, 1024, 2048);
    RoaringBitmapDevice intermediates[] {
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048),
        getIntermediateBitmap(0, 2048)
    };

    int threadsPerBlock = 64;
    int blocksPerGrid = (2048 + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t streams[4];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);

    int numBatches = state.range(0);

    for (auto _ : state)
    {
        for (int i = 0; i < numBatches; i++)
        {
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[0].dev(), 0, 2048);
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[1].dev(), 0, 2048);
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[2].dev(), 0, 2048);
            bitmapUnionNoAlloc<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(a.dev(), b.dev(), intermediates[3].dev(), 0, 2048);
        }
        cudaDeviceSynchronize();
    }
    
    bool output;
    bool* outValue;
    cudaMalloc((void**)&outValue, sizeof(bool));
    bitmapGetBit<<<1, 1>>>(intermediates[0].dev(), 102, outValue);
    cudaMemcpy(&output, outValue, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(outValue);
    
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cudaStreamDestroy(streams[3]);
}
BENCHMARK(BitmapUnionPreAllocateSingleStream)->DenseRange(1, 3, 1);


// BENCHMARK_MAIN();
int main(int argc, char** argv)
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (4096 * 1024 * 1024UL));
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}