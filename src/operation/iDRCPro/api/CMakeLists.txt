add_library(idrc_pro_api
    idrc_api.cpp
)

target_link_libraries(idrc_pro_api
    PUBLIC
        idrc_pro_src
        idrc_pro_config
        idrc_pro_data
        idrc_pro_dm
        idrc_pro_violation
        idrc_pro_rule_stratagy
        idrc_pro_rule_builder
)

target_include_directories(idrc_pro_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)

set(CMAKE_BUILD_TYPE "debug")