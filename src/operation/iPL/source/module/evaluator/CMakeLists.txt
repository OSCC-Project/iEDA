# set
set(iPL_WIRELENGTH_EVALUATOR ${iPL_EVAL}/wirelength)
set(iPL_DENSITY_EVALUATOR ${iPL_EVAL}/density)

add_subdirectory(${iPL_WIRELENGTH_EVALUATOR})
add_subdirectory(${iPL_DENSITY_EVALUATOR})

add_library(ipl-module-evaluator INTERFACE)

target_link_libraries(ipl-module-evaluator
    INTERFACE
        ipl_module_evaluator_wirelength
        ipl_module_evaluator_density
)

target_include_directories(ipl-module-evaluator INTERFACE ${iPL_EVAL})