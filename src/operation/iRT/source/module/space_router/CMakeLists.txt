if(DEBUG_IRT_SPACE_ROUTER)
    message(STATUS "RT: DEBUG_IRT_SPACE_ROUTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_SPACE_ROUTER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_space_router
    ## module space_router
    ${IRT_MODULE}/space_router/SpaceRouter.cpp
)

target_link_libraries(irt_space_router
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_space_router
    PUBLIC
    ${IRT_MODULE}/space_router/sr_data_manager
    ${IRT_MODULE}/space_router
)
