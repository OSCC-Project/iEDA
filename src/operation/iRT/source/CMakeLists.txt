## set path 
set(IRT_DATA_MANAGER ${IRT_SOURCE}/data_manager)
set(IRT_MODULE ${IRT_SOURCE}/module)
set(IRT_SOLVER ${IRT_SOURCE}/solver)
set(IRT_UTILITY ${IRT_SOURCE}/utility)

## build
add_subdirectory(${IRT_DATA_MANAGER})
add_subdirectory(${IRT_MODULE})
add_subdirectory(${IRT_SOLVER})
add_subdirectory(${IRT_UTILITY})

if(DEBUG_IRT_RT)            
    message(STATUS "RT: DEBUG_IRT_RT")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_RT")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_source
    ${IRT_SOURCE}/RT.cpp
)

target_link_libraries(irt_source
    PUBLIC
        irt_data_manager
        irt_module
        irt_solver
        irt_utility
)

target_include_directories(irt_source
    PUBLIC
        ${IRT_SOURCE}
)
