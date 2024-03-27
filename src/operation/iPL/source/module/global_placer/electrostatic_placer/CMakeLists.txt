add_library(ipl-electrostatic_placer NesterovPlace.cc)

target_link_libraries(ipl-electrostatic_placer
    PRIVATE
        ipl_module_evaluator_wirelength
        ipl_module_evaluator_density
        ipl_module_evaluator_timing
        ipl-module-topology_manager
        ipl-module-grid_manager
        ipl-configurator
        ipl-module-evaluator
        ipl-solver
        ipl-bridge
        ipl-utility
)

target_link_libraries(ipl-electrostatic_placer
    PRIVATE
        ipl-api_external_libs
        tool_api_ipl
        tool_manager
)

target_include_directories(ipl-electrostatic_placer PUBLIC ${iPL_ELECTROSTATIC_PLACER})
