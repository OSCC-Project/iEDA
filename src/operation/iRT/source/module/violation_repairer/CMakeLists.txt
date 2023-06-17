if(DEBUG_IRT_VIOLATION_REPAIRER)
    message(STATUS "RT: DEBUG_IRT_VIOLATION_REPAIRER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_VIOLATION_REPAIRER")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(irt_violation_repairer
    ## module violation_repairer
    ${IRT_MODULE}/violation_repairer/ViolationRepairer.cpp
)

target_link_libraries(irt_violation_repairer 
    PUBLIC
        irt_data_manager
        irt_module
        irt_solver
        irt_utility
)

target_include_directories(irt_violation_repairer 
    PUBLIC
        ${IRT_MODULE}/violation_repairer/vr_data_manager
        ${IRT_MODULE}/violation_repairer
)
