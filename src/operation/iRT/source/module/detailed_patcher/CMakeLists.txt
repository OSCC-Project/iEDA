if(DEBUG_IRT_DETAILED_PATCHER)
    message(STATUS "RT: DEBUG_IRT_DETAILED_PATCHER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_DETAILED_PATCHER")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(irt_detailed_patcher
    ## module detailed_patcher
    ${IRT_MODULE}/detailed_patcher/DetailedPatcher.cpp
)

target_link_libraries(irt_detailed_patcher 
    PUBLIC
        irt_data_manager
        irt_module
        irt_toolkit
)

target_include_directories(irt_detailed_patcher 
    PUBLIC
        ${IRT_MODULE}/detailed_patcher/dp_data_manager
        ${IRT_MODULE}/detailed_patcher
)
