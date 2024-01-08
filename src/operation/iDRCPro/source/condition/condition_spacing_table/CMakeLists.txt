add_library(idrc_rule_condition_spacing_table
condition_spacing_table.cpp
)

target_link_libraries(idrc_rule_condition_spacing_table
    PUBLIC
    idrc_rule_condition_matrix
    idrc_pro_engine
    idrc_pro_rule_stratagy
)

target_include_directories(idrc_rule_condition_spacing_table
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_BUILD_TYPE "debug")