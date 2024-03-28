add_subdirectory(${IRT_MODULE}/gds_plotter)
add_subdirectory(${IRT_MODULE}/pin_accessor)
add_subdirectory(${IRT_MODULE}/supply_analyzer)
add_subdirectory(${IRT_MODULE}/planar_router)
add_subdirectory(${IRT_MODULE}/layer_assigner)
add_subdirectory(${IRT_MODULE}/initial_router)
add_subdirectory(${IRT_MODULE}/global_router)
add_subdirectory(${IRT_MODULE}/track_assigner)
add_subdirectory(${IRT_MODULE}/detailed_router)

add_library(irt_module INTERFACE)

target_link_libraries(irt_module 
    INTERFACE
        irt_gds_plotter
        irt_pin_accessor
        irt_supply_analyzer
        irt_planar_router
        irt_layer_assigner
        irt_initial_router
        irt_global_router
        irt_track_assigner
        irt_detailed_router
)
