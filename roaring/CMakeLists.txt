add_library(roaring
    container.cu
    roaring.cu
    roaring_helper.cu
)

set_target_properties(roaring PROPERTIES 
    CUDA_ARCHITECTURES "89"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE TRUE
    )

target_include_directories(roaring INTERFACE
    ${CUDA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS})

target_link_directories(roaring PUBLIC 
    ${CUDA_LIBRARY_DIRS})

target_link_libraries(roaring PUBLIC
    ${CUDA_cublas_LIBRARY}
    CUDA::curand
    CUDA::toolkit
    )

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(roaring PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()