add_subdirectory(engine_init_def)
add_subdirectory(engine_init_rt)

add_library(idrc_engine_init
engine_init.cpp
)

target_link_libraries(idrc_engine_init
    PUBLIC
    idm
    idb
    idrc_pro_dm
    idrc_pro_tech_rules
    idrc_engine_manager    
    solver_geometry_boost
    idrc_engine_init_def
    idrc_engine_init_rt
)

target_include_directories(idrc_engine_init
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_init_rt
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_init_def
    ${HOME_DATABASE}/interaction/RT_DRC
)

if(DEBUG_IDRC_ENGINE_INIT)
  message(STATUS "iDRC: DEBUG_IDRC_ENGINE_INIT")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_ENGINE_INIT")
  set(CMAKE_BUILD_TYPE "Release")
endif()