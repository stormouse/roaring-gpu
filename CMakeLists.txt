cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(roaring-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(roaring-gpu
    main.cu
    container.cu
    roaring.cu
    bitmap_example.cu
    )

set_target_properties(roaring-gpu PROPERTIES 
    CUDA_ARCHITECTURES "89"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE TRUE
    )

target_include_directories(roaring-gpu INTERFACE
    ${CUDA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS})

target_link_directories(roaring-gpu PUBLIC 
    ${CUDA_LIBRARY_DIRS})

target_link_libraries(roaring-gpu PUBLIC
    ${CUDA_cublas_LIBRARY}
    CUDA::curand
    CUDA::toolkit
    )

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(roaring-gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

enable_testing()

add_executable(
    roaring-test
    bitop_ut.cu
    container.cu
    roaring.cu
)
target_link_libraries(
    roaring-test
    GTest::gtest_main
)
set_target_properties(roaring-test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE TRUE
)
include(GoogleTest)
gtest_discover_tests(roaring-test)