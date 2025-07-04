if(DEBUG_IRT_PIN_ACCESSOR)
    message(STATUS "RT: DEBUG_IRT_PIN_ACCESSOR")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_PIN_ACCESSOR")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_pin_accessor
    ## module pin_accessor
    ${IRT_MODULE}/pin_accessor/PinAccessor.cpp
)

target_link_libraries(irt_pin_accessor
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_pin_accessor
    PUBLIC
    ${IRT_MODULE}/pin_accessor/pa_data_manager
    ${IRT_MODULE}/pin_accessor

)
