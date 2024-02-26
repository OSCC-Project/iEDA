file(GLOB_RECURSE DRC_RULE_AREA "*.cpp")

add_library(idrc_pro_rule_area ${DRC_RULE_AREA})

target_link_libraries(idrc_pro_rule_area
    PUBLIC
    idrc_pro_rule_basic
    idrc_condition_manager
        
)

target_include_directories(idrc_pro_rule_area
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)