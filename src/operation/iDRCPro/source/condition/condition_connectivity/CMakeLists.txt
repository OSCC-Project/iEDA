add_library(idrc_rule_condition_connectivity
condition_connectivity.cpp
condition_connectivity_overlap.cpp
)

target_link_libraries(idrc_rule_condition_connectivity
    PUBLIC
    idrc_rule_condition_matrix
    idrc_pro_engine
    idrc_pro_rule_stratagy
)

target_include_directories(idrc_rule_condition_connectivity
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_BUILD_TYPE "debug")