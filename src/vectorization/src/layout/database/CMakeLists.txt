add_library(ivec_layout_db
    vec_layout.cpp
    vec_layer.cpp
    vec_layer_grid.cpp
)

# make sure find out tbb
include_directories(/usr/include)

target_link_libraries(ivec_layout_db
    PUBLIC
    log
    ivec_db
    ivec_util
    tbb
)

target_include_directories(ivec_layout_db
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    Eigen3::Eigen
)