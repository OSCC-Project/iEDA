add_library(ipl-analytical_placer NesterovPlace.cc)

target_link_libraries(ipl-analytical_placer
    PRIVATE
        ipl_module_evaluator_wirelength
        ipl_module_evaluator_density
        ipl-module-topology_manager
        ipl-module-grid_manager
        ipl-configurator
        ipl-module-evaluator
        ipl-solver
        ipl-bridge
        ipl-utility
)

target_link_libraries(ipl-analytical_placer
    PRIVATE
        ipl-api_external_libs
        tool_api_ipl
        tool_manager
)

target_include_directories(ipl-analytical_placer PUBLIC ${iPL_ANALYTICAL_PLACER})
