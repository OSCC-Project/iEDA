add_library(ilm_layout_db
    lm_layout.cpp
    lm_layer.cpp
    lm_layer_grid.cpp
)

target_link_libraries(ilm_layout_db
    PUBLIC
    log
    ilm_db
    ilm_util
)

target_include_directories(ilm_layout_db
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    Eigen3::Eigen
)
set(CMAKE_BUILD_TYPE "debug")
