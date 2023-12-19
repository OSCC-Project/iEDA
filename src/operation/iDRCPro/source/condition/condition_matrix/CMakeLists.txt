add_library(idrc_rule_condition_matrix
condition_matrix.cpp
condition.cpp
)

target_link_libraries(idrc_rule_condition_matrix
    PUBLIC
    idrc_pro_data
    idrc_pro_engine
    idrc_pro_rule_stratagy
)

target_include_directories(idrc_rule_condition_matrix
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)

set(CMAKE_BUILD_TYPE "debug")