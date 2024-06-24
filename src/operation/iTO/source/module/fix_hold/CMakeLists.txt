add_library(ito_fix_hold
HoldOptimizer.cpp
HoldOptimizer_buffers.cpp
HoldOptimizer_init.cpp
HoldOptimizer_process.cpp
)

target_link_libraries(ito_fix_hold
    PUBLIC
    ito_utility
    ito_timing_engine
    ito_config
    ito_eval
    ito_data_manager
    ito_placer

)

target_include_directories(ito_fix_hold
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)


