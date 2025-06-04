if(NOT DEFINED CMD_BUILD)
    # set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

add_subdirectory(${EVAL_MODULE}/congestion)
add_subdirectory(${EVAL_MODULE}/density)
add_subdirectory(${EVAL_MODULE}/timing)
add_subdirectory(${EVAL_MODULE}/wirelength)
add_subdirectory(${EVAL_MODULE}/eval_io)

add_library(eval_module INTERFACE)

target_link_libraries(eval_module 
    INTERFACE
        eval_congestion_eval
        eval_density_eval
        eval_timing_eval
        eval_wirelength_eval
        eval_io
)
