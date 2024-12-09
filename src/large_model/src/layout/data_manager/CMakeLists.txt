add_library(ilm_layout_dm
    lm_layout_dm.cpp
    lm_layout_init.cpp
    lm_wire_pattern.cc
    lm_pattern_gen.cc
    lm_pattern_op.cc
)

find_package(OpenCV REQUIRED)

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
    ${OpenCV_LIBS}
)

target_include_directories(ilm_layout_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
    ${OpenCV_INCLUDE_DIRS}
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE "debug")
