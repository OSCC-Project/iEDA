if(DEBUG_IRT_EARLY_ROUTER)
    message(STATUS "RT: DEBUG_IRT_EARLY_ROUTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_EARLY_ROUTER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_early_router
    ## module early_router
    ${IRT_MODULE}/early_router/EarlyRouter.cpp
)

target_link_libraries(irt_early_router
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_early_router
    PUBLIC
    ${IRT_MODULE}/early_router/er_data_manager
    ${IRT_MODULE}/early_router
)
