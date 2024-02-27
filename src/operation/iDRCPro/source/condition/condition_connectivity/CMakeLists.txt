add_library(idrc_rule_condition_connectivity
condition_connectivity.cpp
condition_connectivity_overlap.cpp
)

target_link_libraries(idrc_rule_condition_connectivity
    PUBLIC
    idrc_rule_condition_matrix
    idrc_pro_engine
    idrc_pro_rule_stratagy
)

target_include_directories(idrc_rule_condition_connectivity
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_CONDITION_CONNECTIVITY)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION_CONNECTIVITY")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION_CONNECTIVITY")
  set(CMAKE_BUILD_TYPE "Release")
endif()
