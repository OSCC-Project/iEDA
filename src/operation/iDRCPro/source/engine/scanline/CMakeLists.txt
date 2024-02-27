add_subdirectory(data_manager)

add_library(idrc_engine_scanline 
engine_scanline.cpp
)

target_link_libraries(idrc_engine_scanline
    PUBLIC
    idrc_engine_scanline_dm    
)

target_include_directories(idrc_engine_scanline
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/database
)

if(DEBUG_IDRC_ENGINE_SCANLINE)
  message(STATUS "iDRC: DEBUG_IDRC_ENGINE_SCANLINE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_ENGINE_SCANLINE")
  set(CMAKE_BUILD_TYPE "Release")
endif()