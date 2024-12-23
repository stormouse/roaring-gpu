add_executable(
    roaring-benchmark
    benchmark.cpp)

target_include_directories(
    roaring-benchmark PUBLIC
    "${benchmark_INCLUDE_DIRS}")

target_link_libraries(
    roaring-benchmark PRIVATE
    benchmark::benchmark)