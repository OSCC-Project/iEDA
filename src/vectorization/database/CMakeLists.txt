add_library(ivec_db
    vec_node.cpp
    vec_net.cpp
    vec_cell.cpp
    vec_instance.cpp
)

target_link_libraries(ivec_db
    PUBLIC
    log
    ivec_util
)

target_include_directories(ivec_db
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
