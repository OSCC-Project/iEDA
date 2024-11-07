add_library(ilm_layout_dm
lm_layout_dm.cpp
lm_layout_init.cpp
)

target_link_libraries(ilm_layout_dm
    PUBLIC
    ilm_layout_db
    idb
    idm
)

target_include_directories(ilm_layout_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)