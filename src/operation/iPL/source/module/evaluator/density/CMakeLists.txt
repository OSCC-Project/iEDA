add_library(ipl_module_evaluator_density
    ${HOME_THIRDPARTY}/fft/fft.cpp
    ${HOME_THIRDPARTY}/fft/fftsg.cpp
    ${HOME_THIRDPARTY}/fft/fftsg2d.cpp
    Density.cc
    ElectricFieldGradient.cc
)

target_link_libraries(ipl_module_evaluator_density
    PRIVATE
    ipl-module-grid_manager
    ipl-utility
)

target_include_directories(ipl_module_evaluator_density 
    PUBLIC 
    ${iPL_DENSITY_EVALUATOR}
    ${HOME_THIRDPARTY}/fft
)