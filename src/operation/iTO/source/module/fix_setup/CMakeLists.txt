add_library(ito_fix_setup
BufferedOption.cpp
SetupOptimizer.cpp
SetupOptimizer_buffers.cpp
SetupOptimizer_init.cpp
SetupOptimizer_process.cpp
SetupOptimizer_gate_sizing.cpp
)

target_link_libraries(ito_fix_setup
    PUBLIC
    ito_utility
    ito_timing_engine
    ito_config
    ito_eval
    ito_data_manager
    ito_fix_drv
    ito_placer

)

target_include_directories(ito_fix_setup
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)


