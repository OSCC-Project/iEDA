add_subdirectory(data_manager)
add_subdirectory(graph)
add_subdirectory(layout)
add_subdirectory(patch)
add_subdirectory(feature)
add_subdirectory(utility)

add_library(vectorization
vectorization.cpp
)

target_link_libraries(vectorization
    PUBLIC
    log
    idm
    ivec_dm
    ivec_feature
    memory
)

target_include_directories(vectorization
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)