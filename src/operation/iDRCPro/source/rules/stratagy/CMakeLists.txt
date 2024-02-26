file(GLOB_RECURSE DRC_RULE_STRATAGY "*.cpp")

add_library(idrc_pro_rule_stratagy ${DRC_RULE_STRATAGY})

target_link_libraries(idrc_pro_rule_stratagy
    PUBLIC
    idrc_pro_data
)

target_include_directories(idrc_pro_rule_stratagy
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)