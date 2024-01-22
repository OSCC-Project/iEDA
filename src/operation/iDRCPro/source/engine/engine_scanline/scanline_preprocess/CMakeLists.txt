file(GLOB_RECURSE DRC_ENGINE_SCANLINE_MANAGER "*.cpp")

add_library(idrc_engine_scanline_preprocess ${DRC_ENGINE_SCANLINE_MANAGER})

target_link_libraries(idrc_engine_scanline_preprocess
    PUBLIC
    solver_geometry_boost
    idrc_pro_data
    idrc_pro_dm
)

target_include_directories(idrc_engine_scanline_preprocess
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(DEBUG_IDRC_ENGINE_SCANLINE)
  set(CMAKE_BUILD_TYPE "Debug")
else()
  set(CMAKE_BUILD_TYPE "Release")
endif()
