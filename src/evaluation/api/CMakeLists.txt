set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

# Congestion Evaluation API
add_library(eval_congestion_api
    ${EVAL_API}/congestion_api.cpp
)

target_link_libraries(eval_congestion_api
    PRIVATE
    eval_db
    eval_congestion_eval
)

target_include_directories(eval_congestion_api
    PUBLIC
    ${EVAL_API}
)

# Density Evaluation API  
add_library(eval_density_api
    ${EVAL_API}/density_api.cpp
)

target_link_libraries(eval_density_api
    PRIVATE
    eval_db
    eval_density_eval
)

target_include_directories(eval_density_api
    PUBLIC
    ${EVAL_API}
)


# Timing & Power Evaluation API  
add_library(eval_timing_api
    ${EVAL_API}/timing_api.cc
)

target_link_libraries(eval_timing_api
    PRIVATE
    eval_db
    eval_timing_eval
)

target_include_directories(eval_timing_api
    PUBLIC
    ${EVAL_API}
    ${EVAL_DATA}
    ${EVAL_MODULE}/timing
)


# Wirelength Evaluation API
add_library(eval_wirelength_api
    ${EVAL_API}/wirelength_api.cpp
)

target_link_libraries(eval_wirelength_api
    PRIVATE
    eval_db
    eval_wirelength_eval
)

target_include_directories(eval_wirelength_api
    PUBLIC
    ${EVAL_API}
)


# union API
add_library(eval_union_api
    ${EVAL_API}/union_api.cpp
)

target_link_libraries(eval_union_api
    PRIVATE
        eval_util_init_flute  
        eval_util_init_egr   
        eval_util_init_idb
)

target_include_directories(eval_union_api
    PUBLIC
        ${EVAL_DATA}
)
