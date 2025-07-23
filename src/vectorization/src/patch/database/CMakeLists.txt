add_library(ivec_patch_db
    vec_patch.cpp
    vec_patch_grid.cpp
)

target_link_libraries(ivec_patch_db
    PUBLIC
    log
    ivec_db
    ivec_util
)

target_include_directories(ivec_patch_db
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)