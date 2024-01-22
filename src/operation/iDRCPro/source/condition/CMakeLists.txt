add_library(idrc_condition_manager
  condition_manager.cpp
)

target_link_libraries(idrc_condition_manager
    PUBLIC
    idrc_engine_check
)

target_include_directories(idrc_condition_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_CONDITION_BUILDER)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION_BUILDER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION_BUILDER")
  set(CMAKE_BUILD_TYPE "Release")
endif()
