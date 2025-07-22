add_library(ilm_util
    lm_grid_info.cpp
)

target_link_libraries(ilm_util
    PUBLIC
)

target_include_directories(ilm_util
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)