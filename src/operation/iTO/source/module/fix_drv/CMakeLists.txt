if(DEBUG_ITO_FIX_DRV)
  message(STATUS "TO: DEBUG_ITO_FIX_DRV")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "TO: RELEASE_ITO_FIX_DRV")
  set(CMAKE_BUILD_TYPE "Release")
endif()

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
    ito_tree_build
)

target_include_directories(ito_fix_drv
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
    ${HOME_OPERATION}/iTO/source/module/solver
)


