file(GLOB_RECURSE DRC_ENGINE_SCANLINE_MANAGER "*.cpp")

add_library(idrc_engine_scanline_dm ${DRC_ENGINE_SCANLINE_MANAGER})

target_link_libraries(idrc_engine_scanline_dm
    PUBLIC
    solver_geometry_boost
)

target_include_directories(idrc_engine_scanline_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../database
)

set(CMAKE_BUILD_TYPE "debug")