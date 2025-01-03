add_executable(
    roaring-benchmark
    benchmark.cu)

target_include_directories(
    roaring-benchmark PUBLIC
    "${benchmark_INCLUDE_DIRS}"
    ${PROJECT_SOURCE_DIR}/roaring)

target_link_libraries(
    roaring-benchmark PRIVATE
    benchmark::benchmark
    roaring)