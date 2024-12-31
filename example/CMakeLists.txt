add_executable(example
    main.cu
)

target_include_directories(
    example PUBLIC
    ${PROJECT_SOURCE_DIR}/roaring
)

target_link_libraries(
    example PRIVATE
    roaring
)