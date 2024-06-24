add_library(ito_fix_drv
ViolationOptimizer.cpp
ViolationOptimizer_init.cpp
ViolationOptimizer_check.cpp
ViolationOptimizer_repair.cpp
ViolationOptimizer_buffers.cpp
)

target_link_libraries(ito_fix_drv
    PUBLIC
    ito_timing_engine
    ito_utility
    ito_eval
    ito_data_manager
    ito_config
    ito_placer

)

target_include_directories(ito_fix_drv
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)


