add_subdirectory(${IRT_MODULE}/drc_engine)
add_subdirectory(${IRT_MODULE}/gds_plotter)
add_subdirectory(${IRT_MODULE}/pin_accessor)
add_subdirectory(${IRT_MODULE}/supply_analyzer)
add_subdirectory(${IRT_MODULE}/topology_generator)
add_subdirectory(${IRT_MODULE}/layer_assigner)
add_subdirectory(${IRT_MODULE}/space_router)
add_subdirectory(${IRT_MODULE}/track_assigner)
add_subdirectory(${IRT_MODULE}/detailed_router)
add_subdirectory(${IRT_MODULE}/violation_reporter)
add_subdirectory(${IRT_MODULE}/early_router)

add_library(irt_module INTERFACE)

target_link_libraries(irt_module
    INTERFACE
    irt_drc_engine
    irt_gds_plotter
    irt_pin_accessor
    irt_supply_analyzer
    irt_topology_generator
    irt_layer_assigner
    irt_space_router
    irt_track_assigner
    irt_detailed_router
    irt_violation_reporter
    irt_early_router
)
