if(DEBUG_IRT_EARLY_GLOBAL_ROUTER)
    message(STATUS "RT: DEBUG_IRT_EARLY_GLOBAL_ROUTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_EARLY_GLOBAL_ROUTER")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(irt_early_global_router
    ## module early_global_router
    ${IRT_MODULE}/early_global_router/egr_data_manager/EGRDataManager.cpp
    ${IRT_MODULE}/early_global_router/EarlyGlobalRouter.cpp
)

target_link_libraries(irt_early_global_router 
    PUBLIC
        irt_data_manager
        irt_module
        irt_utility
)

target_include_directories(irt_early_global_router 
    PUBLIC
        ${IRT_MODULE}/early_global_router/egr_data_manager
        ${IRT_MODULE}/early_global_router
)
