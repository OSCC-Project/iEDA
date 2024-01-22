add_library(idrc_rule_condition_spacing
condition_spacing.cpp
)

target_link_libraries(idrc_rule_condition_spacing
    PUBLIC
    idrc_rule_condition_matrix
    idrc_pro_engine
    idrc_pro_rule_stratagy
)

target_include_directories(idrc_rule_condition_spacing
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_BUILD_TYPE "debug")