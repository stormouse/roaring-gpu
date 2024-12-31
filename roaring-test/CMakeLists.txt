add_executable(
    roaring-test
    bitop.cu
    container.cu
    bitmap.cu
    roaring_gpu.cu
)

target_include_directories(
    roaring-test PUBLIC
    ${PROJECT_SOURCE_DIR}/roaring
)

target_link_libraries(
    roaring-test
    roaring
    GTest::gtest_main
)

set_target_properties(roaring-test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE TRUE
)

enable_testing()
gtest_discover_tests(roaring-test)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE "1000000")