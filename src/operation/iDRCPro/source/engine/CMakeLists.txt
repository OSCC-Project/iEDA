add_subdirectory(engine_manager)
add_subdirectory(engine_init)
add_subdirectory(scanline)

file(GLOB_RECURSE DRC_ENGINE "*.cpp")

add_library(idrc_pro_engine ${DRC_ENGINE})

target_link_libraries(idrc_pro_engine
    PUBLIC
    idrc_pro_dm
    idrc_engine_manager
    idrc_engine_init
)

target_include_directories(idrc_pro_engine
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)

if(DEBUG_IDRC_ENGINE)
  message(STATUS "iDRC: DEBUG_IDRC_ENGINE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_ENGINE")
  set(CMAKE_BUILD_TYPE "Release")
endif()
