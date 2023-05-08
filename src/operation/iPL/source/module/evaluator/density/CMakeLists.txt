add_library(ipl_module_evaluator_density
    fft/fft.cpp
    fft/fftsg.cpp
    fft/fftsg2d.cpp
    Density.cc
    ElectricFieldGradient.cc
)

target_link_libraries(ipl_module_evaluator_density
    PRIVATE
    ipl-module-grid_manager
    ipl-utility
)

target_include_directories(ipl_module_evaluator_density PUBLIC ${iPL_DENSITY_EVALUATOR})