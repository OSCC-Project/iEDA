add_library(ivec_dm
    vec_dm.cpp
    vec_file.cpp
)

target_link_libraries(ivec_dm
    PUBLIC
    ivec_layout_dm
    ivec_layout_db
    ivec_graph_dm
    ivec_patch_dm
    ivec_patch_db
    ivec_util
)

target_include_directories(ivec_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)