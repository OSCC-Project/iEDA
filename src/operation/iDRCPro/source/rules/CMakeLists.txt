
add_library(idrc_pro_tech_rules
    tech_rules.cpp
)

target_link_libraries(idrc_pro_tech_rules 
    PUBLIC
    idm
    idb

)

target_include_directories(idrc_pro_tech_rules 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
