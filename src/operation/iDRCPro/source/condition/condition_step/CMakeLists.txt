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

set(CMAKE_BUILD_TYPE "debug")