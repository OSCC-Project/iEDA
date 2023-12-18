add_subdirectory(database)
add_subdirectory(rule_check)
add_subdirectory(stratagy)
add_subdirectory(rule_builder)
add_subdirectory(tech_rules)

file(GLOB_RECURSE DRC_RULE_MANAGER "*.cpp")

add_library(idrc_pro_rule_manager ${DRC_RULE_MANAGER})

target_link_libraries(idrc_pro_rule_manager
    PUBLIC
    idrc_pro_tech_rules
    idrc_pro_engine
    idrc_pro_rule_stratagy
    idrc_pro_rule_builder
    idrc_pro_rule_area
    idrc_pro_rule_spacing
)

target_include_directories(idrc_pro_rule_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_BUILD_TYPE "debug")