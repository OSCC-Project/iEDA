if(DEBUG_IRT_PIN_PATCHER)
    message(STATUS "RT: DEBUG_IRT_PIN_PATCHER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_PIN_PATCHER")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(irt_pin_patcher
    ## module pin_patcher
    ${IRT_MODULE}/pin_patcher/PinPatcher.cpp
)

target_link_libraries(irt_pin_patcher 
    PUBLIC
        irt_data_manager
        irt_module
        irt_toolkit
)

target_include_directories(irt_pin_patcher 
    PUBLIC
        ${IRT_MODULE}/pin_patcher/pp_data_manager
        ${IRT_MODULE}/pin_patcher
)
