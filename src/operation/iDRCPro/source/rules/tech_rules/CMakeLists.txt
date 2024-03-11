
add_library(idrc_pro_tech_rules
    tech_rules.cpp
    tech_rule_layer.cpp
    rule_basic.cpp
    rule_condition_map.cpp
    rule_condition_area.cpp
    rule_condition_width.cpp
    rule_condition_spacing.cpp
    rule_condition_edge.cpp
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
  set(CMAKE_BUILD_TYPE "Debug")
else()
  set(CMAKE_BUILD_TYPE "Release")
endif()
