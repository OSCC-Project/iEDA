add_subdirectory(condition_detail_jog)

add_library(idrc_condition_manager
  condition_manager.cpp
)

target_link_libraries(idrc_condition_manager
    PUBLIC
    idm
    idb
    idrc_engine_check
    idrc_pro_violation
    idrc_condition_detail_jog
    idrc_pro_util
)

target_include_directories(idrc_condition_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/condition_detail_jog
)

if(DEBUG_IDRC_CONDITION)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION")
  set(CMAKE_BUILD_TYPE "Release")
endif()
