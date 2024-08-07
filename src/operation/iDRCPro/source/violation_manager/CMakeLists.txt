add_library(idrc_pro_violation
    idrc_violation_manager.cpp
    idrc_violation.cpp
)

target_link_libraries(idrc_pro_violation
    PUBLIC
    solver_geometry_boost
    idrc_pro_tech_rules
    idrc_pro_region_query
    idm
    idrc_engine_manager
)

target_include_directories(idrc_pro_violation
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)
