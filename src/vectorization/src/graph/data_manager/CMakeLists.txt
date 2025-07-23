set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
option(BUILD_VEC_GUI "Enable GUI components (default OFF)" OFF)

if(POLICY CMP0100)
    cmake_policy(SET CMP0100 NEW)
endif()

set(IVEC_GRAPH_DM_SRCS
    vec_graph_dm.cpp
    vec_graph_check.cc
    vec_net_graph_gen.cc
)

if(BUILD_VEC_GUI)
    add_definitions(-DBUILD_VEC_GUI)

    list(APPEND IVEC_GRAPH_DM_SRCS
        vec_graph_gui.cc
        vec_graph_gui.hh
    )

    find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL REQUIRED)
    find_package(OpenGL REQUIRED)
endif()

add_library(ivec_graph_dm ${IVEC_GRAPH_DM_SRCS})

target_link_libraries(ivec_graph_dm
    PUBLIC
    idb
    idm
    geometry_db
    log
    usage
    ivec_db
    ivec_dm
    ivec_layout_db
    ivec_util
    eval_timing_api
)

if(BUILD_VEC_GUI)
    target_link_libraries(ivec_graph_dm
        PUBLIC
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        ${OPENGL_LIBRARIES}
    )
endif()

target_include_directories(ivec_graph_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})