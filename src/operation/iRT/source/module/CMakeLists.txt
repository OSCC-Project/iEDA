add_subdirectory(${IRT_MODULE}/detailed_router)
add_subdirectory(${IRT_MODULE}/early_global_router)
add_subdirectory(${IRT_MODULE}/gds_plotter)
add_subdirectory(${IRT_MODULE}/global_router)
add_subdirectory(${IRT_MODULE}/pin_accessor)
add_subdirectory(${IRT_MODULE}/supply_analyzer)
add_subdirectory(${IRT_MODULE}/track_assigner)

add_library(irt_module INTERFACE)

target_link_libraries(irt_module 
    INTERFACE
        irt_detailed_router
        irt_early_global_router
        irt_gds_plotter
        irt_global_router
        irt_pin_accessor
        irt_supply_analyzer
        irt_track_assigner
)
