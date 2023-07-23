add_subdirectory(dct_process)

add_library(ipl_module_evaluator_density
    Density.cc
    ElectricFieldGradient.cc
)

target_link_libraries(ipl_module_evaluator_density
    PRIVATE
    ipl-dct
    ipl-module-grid_manager
)

target_include_directories(ipl_module_evaluator_density 
    PUBLIC 
    ${iPL_DENSITY_EVALUATOR}
)