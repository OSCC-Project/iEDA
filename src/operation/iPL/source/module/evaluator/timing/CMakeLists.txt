add_library(ipl_module_evaluator_timing
    TimingAnnotation.cc
)

target_link_libraries(ipl_module_evaluator_timing
    PUBLIC
        ipl-module-topology_manager
        ipl_module_evaluator_wirelength
    PRIVATE
        ipl-module-logger
        ipl-configurator
        ipl-bridge        
)

target_include_directories(ipl_module_evaluator_timing PUBLIC ${iPL_TIMING_EVALUATOR})