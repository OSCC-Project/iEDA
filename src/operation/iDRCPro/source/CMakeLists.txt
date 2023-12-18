add_subdirectory(config)
add_subdirectory(data)
add_subdirectory(data_manager)
add_subdirectory(violation_manager)
add_subdirectory(engine)
add_subdirectory(region_query)
add_subdirectory(rules)
add_subdirectory(condition)
# add_subdirectory(util)

add_library(idrc_pro_src
    idrc.cpp
)

target_link_libraries(idrc_pro_src
    PUBLIC
        idrc_pro_config
        idrc_pro_data
        idrc_pro_dm
        idrc_pro_engine
        idrc_pro_rule_manager
        # idrc_pro_util
)

target_include_directories(idrc_pro_src
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_BUILD_TYPE "debug")