if(DEBUG_IRT_DRC_CHECKER)            
    message(STATUS "RT: DEBUG_IRT_DRC_CHECKER")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_DRC_CHECKER")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_drc_checker
    ## module drc_checker
    ${IRT_MODULE}/drc_checker/DRCChecker.cpp
)

target_link_libraries(irt_drc_checker 
    PUBLIC
        irt_data_manager
        irt_module
        irt_solver
        irt_utility
)

target_include_directories(irt_drc_checker 
    PUBLIC
        ${IRT_MODULE}/drc_checker/dc_data_manager
        ${IRT_MODULE}/drc_checker
)
