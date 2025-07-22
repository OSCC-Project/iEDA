add_library(ilm_layout_dm
    lm_layout_dm.cpp
    lm_layout_init.cpp
    lm_wire_pattern.cc
    lm_pattern_gen.cc
)

# find_package(OpenCV REQUIRED)

target_link_libraries(ilm_layout_dm
    PUBLIC
    idb
    idm
    geometry_db
    log
    usage
    ilm_db
    ilm_layout_db
    ilm_util
)

target_include_directories(ilm_layout_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})