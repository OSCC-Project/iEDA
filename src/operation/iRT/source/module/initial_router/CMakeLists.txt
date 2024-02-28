if(DEBUG_IRT_INITIAL_ROUTER)
    message(STATUS "RT: DEBUG_IRT_INITIAL_ROUTER")   
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_INITIAL_ROUTER")   
    set(CMAKE_BUILD_TYPE "Release")    
endif()

add_library(irt_initial_router
    ## module initial_router
    ${IRT_MODULE}/initial_router/InitialRouter.cpp
)

target_link_libraries(irt_initial_router 
    PUBLIC
        irt_data_manager
        irt_module
        irt_solver
        irt_utility
)

target_include_directories(irt_initial_router 
    PUBLIC
        ${IRT_MODULE}/initial_router/ir_data_manager
        ${IRT_MODULE}/initial_router
)
