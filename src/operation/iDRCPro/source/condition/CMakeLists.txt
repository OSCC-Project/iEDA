add_library(idrc_condition_manager
  condition_manager.cpp
  condition_min_spacing.cpp
  condition_overlap.cpp
  condition_wire_width.cpp
  condition_polygon.cpp
)

target_link_libraries(idrc_condition_manager
    PUBLIC
    idrc_pro_util
    idrc_pro_violation
    idrc_pro_tech_rules
    idrc_engine_manager
    idrc_pro_data
)

target_include_directories(idrc_condition_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_CONDITION)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION")
  set(CMAKE_BUILD_TYPE "Release")
endif()
