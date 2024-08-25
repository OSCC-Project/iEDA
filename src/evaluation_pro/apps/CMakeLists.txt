# SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

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
