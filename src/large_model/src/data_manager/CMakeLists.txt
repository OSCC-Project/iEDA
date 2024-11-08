add_library(ilm_dm
    lm_dm.cpp
)

target_link_libraries(ilm_dm
    PUBLIC
    ilm_layout_dm
    ilm_layout_db
)

target_include_directories(ilm_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set(CMAKE_BUILD_TYPE "debug")