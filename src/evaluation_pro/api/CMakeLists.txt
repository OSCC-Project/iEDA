set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

# Congestion Evaluation API
add_library(eval_pro_congestion_api
    ${EVAL_PRO_API}/congestion_api.cpp
)

target_link_libraries(eval_pro_congestion_api
    PRIVATE
    eval_pro_db
    eval_pro_congestion_eval
)

target_include_directories(eval_pro_congestion_api
    PUBLIC
    ${EVAL_PRO_API}
)

# Density Evaluation API  
add_library(eval_pro_density_api
    ${EVAL_PRO_API}/density_api.cpp
)

target_link_libraries(eval_pro_density_api
    PRIVATE
    eval_pro_db
    eval_pro_density_eval
)

target_include_directories(eval_pro_density_api
    PUBLIC
    ${EVAL_PRO_API}
)


# Timing & Power Evaluation API  
add_library(eval_pro_timing_api
    ${EVAL_PRO_API}/timing_api.cc
)

target_link_libraries(eval_pro_timing_api
    PRIVATE
    eval_pro_db
    eval_pro_timing_eval
)

target_include_directories(eval_pro_timing_api
    PUBLIC
    ${EVAL_PRO_API}
    ${EVAL_PRO_DATA}
    ${EVAL_PRO_MODULE}/timing
)


# Wirelength Evaluation API
add_library(eval_pro_wirelength_api
    ${EVAL_PRO_API}/wirelength_api.cpp
)

target_link_libraries(eval_pro_wirelength_api
    PRIVATE
    eval_pro_db
    eval_pro_wirelength_eval
)

target_include_directories(eval_pro_wirelength_api
    PUBLIC
    ${EVAL_PRO_API}
)
