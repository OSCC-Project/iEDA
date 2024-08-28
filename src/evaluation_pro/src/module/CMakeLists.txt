# SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

add_subdirectory(${EVAL_PRO_MODULE}/congestion)
add_subdirectory(${EVAL_PRO_MODULE}/wirelength)
add_subdirectory(${EVAL_PRO_MODULE}/density)


add_library(eval_pro_module INTERFACE)

target_link_libraries(eval_pro_module 
    INTERFACE
        eval_pro_congestion_eval
        eval_pro_wirelength_eval
        eval_pro_density_eval
)
