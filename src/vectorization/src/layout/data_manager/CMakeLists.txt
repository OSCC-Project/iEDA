add_library(ivec_layout_dm
    vec_layout_dm.cpp
    vec_layout_init.cpp
    vec_wire_pattern.cc
    vec_pattern_gen.cc
)

# find_package(OpenCV REQUIRED)

target_link_libraries(ivec_layout_dm
    PUBLIC
    idb
    idm
    geometry_db
    log
    usage
    ivec_db
    ivec_layout_db
    ivec_util
)

target_include_directories(ivec_layout_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})