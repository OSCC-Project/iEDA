add_library(idrc_rule_condition_step
condition_step.cpp
)

target_link_libraries(idrc_rule_condition_step
    PUBLIC
    idrc_rule_condition_matrix
    idrc_pro_engine
    idrc_pro_rule_stratagy
)

target_include_directories(idrc_rule_condition_step
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_CONDITION_STEP)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION_STEP")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION_STEP")
  set(CMAKE_BUILD_TYPE "Release")
endif()
