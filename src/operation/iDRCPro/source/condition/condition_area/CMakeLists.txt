add_library(idrc_rule_condition_area
condition_area.cpp
)

target_link_libraries(idrc_rule_condition_area
    PUBLIC
    idrc_rule_condition_matrix
    idrc_pro_engine
    idrc_pro_rule_stratagy
    solver_geometry_boost
)

target_include_directories(idrc_rule_condition_area
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_CONDITION_AREA)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION_AREA")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION_AREA")
  set(CMAKE_BUILD_TYPE "Release")
endif()