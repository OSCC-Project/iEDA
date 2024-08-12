add_library(idrc_engine_init_def

engine_init_def.cpp

)

target_link_libraries(idrc_engine_init_def
    PUBLIC
    idm
    idb
    idrc_pro_dm
    idrc_pro_tech_rules
    idrc_engine_manager    
    solver_geometry_boost
    idrc_pro_data
)

target_include_directories(idrc_engine_init_def
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)

if(DEBUG_IDRC_ENGINE_INIT)
  set(CMAKE_BUILD_TYPE "Debug")
else()
  set(CMAKE_BUILD_TYPE "Release")
endif()