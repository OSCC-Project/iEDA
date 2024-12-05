add_library(ilm_graph_dm
    lm_graph_dm.cpp
    lm_graph_check.cc
    lm_net_graph_gen.cc
)

find_package(OpenCV REQUIRED)

target_link_libraries(ilm_graph_dm
    PUBLIC
    idb
    idm
    geometry_db
    log
    usage
    ilm_db
    ilm_dm
    ilm_layout_db
    ilm_util
)

target_include_directories(ilm_graph_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE "debug")
