file(GLOB_RECURSE DRC_UTIL "*.cpp")

add_library(idrc_pro_util ${DRC_UTIL})

target_link_libraries(idrc_pro_util
    PUBLIC
    solver_geometry_boost
)

target_include_directories(idrc_pro_util
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)