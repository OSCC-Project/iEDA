add_library(ivec_util
    vec_grid_info.cpp
)

target_link_libraries(ivec_util
    PUBLIC
)

target_include_directories(ivec_util
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)