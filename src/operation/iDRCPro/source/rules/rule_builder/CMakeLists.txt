add_library(idrc_pro_rule_builder
    rule_builder.cpp
)

target_link_libraries(idrc_pro_rule_builder
    PUBLIC
    idm
    idb
    idrc_pro_tech_rules
)

target_include_directories(idrc_pro_rule_builder
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)