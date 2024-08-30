# SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

# 拥塞评估app
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

# 密度评估app
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


# 线长评估app
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


