# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

# Congestion Evaluation APP
add_executable(congestion_app
    congestion_app.cpp
)

target_link_libraries(congestion_app
    PRIVATE
    eval_pro_congestion_api
)

target_include_directories(congestion_app
    PRIVATE
    ${HOME_EVALUATION_PRO}/api
    ${HOME_EVALUATION_PRO}/database
)

# Density Evaluation APP
add_executable(density_app
    density_app.cpp
)

target_link_libraries(density_app
    PRIVATE
    eval_pro_density_api
)

target_include_directories(density_app
    PRIVATE
    ${HOME_EVALUATION_PRO}/api
    ${HOME_EVALUATION_PRO}/database
)

# Timing & Power Evaluation APP
add_executable(timing_app
    timing_app.cc
)

target_link_libraries(timing_app
    PRIVATE
    eval_pro_timing_api
    idm
    ipl-api
    ipl-external_api
)

target_include_directories(timing_app
    PRIVATE
    ${HOME_EVALUATION_PRO}/api
    ${HOME_EVALUATION_PRO}/database
)

# Wirelength Evaluation APP
add_executable(wirelength_app
    wirelength_app.cpp
)

target_link_libraries(wirelength_app
    PRIVATE
    eval_pro_wirelength_api
)

target_include_directories(wirelength_app
    PRIVATE
    ${HOME_EVALUATION_PRO}/api
    ${HOME_EVALUATION_PRO}/database
)

