add_library(ilm_patch_db
    lm_patch.cpp
    lm_patch_grid.cpp
)

target_link_libraries(ilm_patch_db
    PUBLIC
    log
    ilm_db
    ilm_util
)

target_include_directories(ilm_patch_db
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)