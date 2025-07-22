add_library(ilm_dm
    lm_dm.cpp
    lm_file.cpp
)

target_link_libraries(ilm_dm
    PUBLIC
    ilm_layout_dm
    ilm_layout_db
    ilm_graph_dm
    ilm_patch_dm
    ilm_patch_db
    ilm_util
)

target_include_directories(ilm_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)