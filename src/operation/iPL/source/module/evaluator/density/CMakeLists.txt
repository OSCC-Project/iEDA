add_subdirectory(dct_process)

add_library(ipl_module_evaluator_density
    Density.cc
    ElectricFieldGradient.cc
)
set_target_properties(ipl_module_evaluator_density PROPERTIES LINK_INTERFACE_MULTIPLICITY 3)
get_property(LIM_COUNT TARGET ipl_module_evaluator_density PROPERTY "LINK_INTERFACE_MULTIPLICITY")
message("LINK_INTERFACE_MULTIPLICITY: ${LIM_COUNT}")

target_link_libraries(ipl_module_evaluator_density
    PRIVATE
    ipl-dct
    ipl-module-grid_manager
)

target_include_directories(ipl_module_evaluator_density 
    PUBLIC 
    ${iPL_DENSITY_EVALUATOR}
)