file(GLOB_RECURSE DRC_ENGINE_CHECK "*.cpp")

add_library(idrc_engine_check ${DRC_ENGINE_CHECK})

target_link_libraries(idrc_engine_check
    PUBLIC
    idrc_condition_manager
)

target_include_directories(idrc_engine_check
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_ENGINE_CHECK)
  message(STATUS "iDRC: DEBUG_IDRC_ENGINE_CHECK")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_ENGINE_CHECK")
  set(CMAKE_BUILD_TYPE "Release")
endif()
