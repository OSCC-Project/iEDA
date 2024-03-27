if(DEBUG_IRT_PLANAR_ROUTER)
    message(STATUS "RT: DEBUG_IRT_PLANAR_ROUTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_PLANAR_ROUTER")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(irt_planar_router
    ## module planar_router
    ${IRT_MODULE}/planar_router/PlanarRouter.cpp
)

target_link_libraries(irt_planar_router 
    PUBLIC
        irt_data_manager
        irt_module
        irt_utility
)

target_include_directories(irt_planar_router 
    PUBLIC
        ${IRT_MODULE}/planar_router/pr_data_manager
        ${IRT_MODULE}/planar_router
        
)
