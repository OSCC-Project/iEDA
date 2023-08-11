add_subdirectory(${IRT_MODULE}/detailed_router)
add_subdirectory(${IRT_MODULE}/drc_checker)
add_subdirectory(${IRT_MODULE}/early_global_router)
add_subdirectory(${IRT_MODULE}/gds_plotter)
add_subdirectory(${IRT_MODULE}/global_router)
add_subdirectory(${IRT_MODULE}/pin_accessor)
add_subdirectory(${IRT_MODULE}/resource_allocator)
add_subdirectory(${IRT_MODULE}/track_assigner)
add_subdirectory(${IRT_MODULE}/violation_repairer)

add_library(irt_module INTERFACE)

target_link_libraries(irt_module 
    INTERFACE
        irt_detailed_router
        irt_drc_checker
        irt_early_global_router
        irt_gds_plotter
        irt_global_router
        irt_pin_accessor
        irt_resource_allocator
        irt_track_assigner
        irt_violation_repairer
)
