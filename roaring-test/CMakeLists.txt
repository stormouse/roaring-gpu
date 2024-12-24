add_executable(
    roaring-test
    bitop_test.cu
    bitmap_test.cu
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