cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(roaring-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(roaring)

# ---- Testing ---- #

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

add_subdirectory(roaring-test)

# ---- Benchmark ---- #

find_package(benchmark REQUIRED)
add_subdirectory(roaring-benchmark)