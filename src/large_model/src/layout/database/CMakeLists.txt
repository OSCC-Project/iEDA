add_library(ilm_layout_db
lm_layout.cpp
lm_layer.cpp
lm_layer_grid.cpp
lm_patch.cpp
lm_node.cpp
lm_net.cpp
)

target_include_directories(ilm_layout_db 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set(CMAKE_BUILD_TYPE "debug")