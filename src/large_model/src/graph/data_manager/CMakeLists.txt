set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
option(BUILD_LM_GUI "Enable GUI components (default OFF)" OFF)

add_library(ilm_graph_dm
    lm_graph_dm.cpp
    lm_graph_check.cc
    lm_net_graph_gen.cc
)

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
    eval_timing_api
)

target_include_directories(ilm_graph_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

if(BUILD_LM_GUI)
add_definitions(-DBUILD_LM_GUI)
add_library(ilm_graph_dm
    lm_graph_gui.cc
    lm_graph_gui.hh
)
find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL REQUIRED)
find_package(OpenGL REQUIRED)

target_link_libraries(ilm_graph_dm
    PUBLIC
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::OpenGL
    ${OPENGL_LIBRARIES}
)
endif()

set(CMAKE_BUILD_TYPE "debug")
