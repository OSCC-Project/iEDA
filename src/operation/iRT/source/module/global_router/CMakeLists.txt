if(DEBUG_IRT_GLOBAL_ROUTER)
    message(STATUS "RT: DEBUG_IRT_GLOBAL_ROUTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_GLOBAL_ROUTER")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(irt_global_router
    ## module global_router
    ${IRT_MODULE}/global_router/GlobalRouter.cpp
)

target_link_libraries(irt_global_router 
    PUBLIC
        irt_data_manager
        irt_module
        irt_toolkit
)

target_include_directories(irt_global_router 
    PUBLIC
        ${IRT_MODULE}/global_router/gr_data_manager
        ${IRT_MODULE}/global_router
        
)
