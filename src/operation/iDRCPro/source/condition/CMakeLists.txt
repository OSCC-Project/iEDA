add_subdirectory(condition_matrix)
add_subdirectory(condition_area)
add_subdirectory(condition_spacing)
add_subdirectory(condition_connectivity)
add_subdirectory(condition_step)
add_subdirectory(condition_eol)

add_library(idrc_condition_manager
    condition_manager.cpp
    condition_builder.cpp
    condition_builder_routing.cpp
)

target_link_libraries(idrc_condition_manager
    PUBLIC
    idrc_pro_engine
    idrc_pro_rule_stratagy
    idrc_rule_condition_matrix
    idrc_rule_condition_area
    idrc_rule_condition_connectivity
    idrc_rule_condition_spacing
    idrc_rule_condition_step
    idrc_rule_condition_eol
    idrc_pro_tech_rules
    idrc_pro_violation
    idrc_pro_util
)

target_include_directories(idrc_condition_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_BUILD_TYPE "debug")