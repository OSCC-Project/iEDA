
add_library(idrc_pro_rule_spacing
    rule_spacing.cpp
)

target_link_libraries(idrc_pro_rule_spacing 
    PUBLIC
    idrc_pro_rule_basic
    idrc_condition_manager

)

target_include_directories(idrc_pro_rule_spacing 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
