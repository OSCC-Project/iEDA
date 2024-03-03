add_library(ipl_module_evaluator_wirelength
    HPWirelength.cc
    WAWirelengthGradient.cc
    SteinerWirelength.cc
)


target_link_libraries(ipl_module_evaluator_wirelength
    PUBLIC
        ipl-module-topology_manager
        ipl-module-grid_manager 
    PRIVATE
        ipl-module-logger
        ipl-configurator
        ipl-bridge
)

target_include_directories(ipl_module_evaluator_wirelength PUBLIC ${iPL_WIRELENGTH_EVALUATOR})