add_library(idrc_condition_detail_jog
  condition_detail_jog.cpp
)

target_link_libraries(idrc_condition_detail_jog
    PUBLIC
    idm
    idb
    idrc_engine_check
    idrc_pro_violation
    idrc_condition_manager
)

target_include_directories(idrc_condition_detail_jog
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_CONDITION_JOG)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION_JOG")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION_JOG")
  set(CMAKE_BUILD_TYPE "Release")
endif()
