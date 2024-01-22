
add_library(idrc_pro_tech_rules
    tech_rules.cpp
    rule_builder.cpp
)

target_link_libraries(idrc_pro_tech_rules 
    PUBLIC
    idm
    idb

)

target_include_directories(idrc_pro_tech_rules 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_RULE)
  message(STATUS "iDRC: DEBUG_IDRC_RULE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_RULE")
  set(CMAKE_BUILD_TYPE "Release")
endif()
