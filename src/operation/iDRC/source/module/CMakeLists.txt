add_subdirectory(${IDRC_MODULE}/gds_plotter)
add_subdirectory(${IDRC_MODULE}/rule_validator)

add_library(idrc_module INTERFACE)

target_link_libraries(idrc_module
    INTERFACE
    idrc_gds_plotter
    idrc_rule_validator
)
