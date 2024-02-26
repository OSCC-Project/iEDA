
add_library(idrc_pro_rule_basic
    rule_base.cpp
)

target_link_libraries(idrc_pro_rule_basic 
    PUBLIC
    idrc_pro_tech_rules
    idrc_pro_engine
    idrc_condition_manager
)

target_include_directories(idrc_pro_rule_basic 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
