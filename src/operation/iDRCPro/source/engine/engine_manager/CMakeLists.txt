file(GLOB_RECURSE DRC_ENGINE_MANAGER "*.cpp")

add_library(idrc_engine_manager ${DRC_ENGINE_MANAGER})

target_link_libraries(idrc_engine_manager
    PUBLIC
    solver_geometry
    solver_geometry_boost
    idrc_engine_scanline
    idrc_pro_data
    idrc_pro_tech_rules
    idrc_pro_violation
    idm
)

target_include_directories(idrc_engine_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_ENGINE_MANAGER)
  message(STATUS "iDRC: DEBUG_IDRC_ENGINE_MANAGER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_ENGINE_MANAGER")
  set(CMAKE_BUILD_TYPE "Release")
endif()