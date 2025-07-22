add_library(ilm_db
    lm_node.cpp
    lm_net.cpp
)

target_link_libraries(ilm_db
    PUBLIC
    log
    ilm_util
)

target_include_directories(ilm_db
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
