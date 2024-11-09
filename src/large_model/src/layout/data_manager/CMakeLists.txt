add_library(ilm_layout_dm
lm_layout_dm.cpp
lm_layout_init.cpp
)

target_link_libraries(ilm_layout_dm
    PUBLIC
    idb
    idm
    geometry_db
    log
    ilm_layout_db
)

target_include_directories(ilm_layout_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set(CMAKE_BUILD_TYPE "debug")