if(DEBUG_IRT_DETAILED_ROUTER)
    message(STATUS "RT: DEBUG_IRT_DETAILED_ROUTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_DETAILED_ROUTER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_detailed_router
    ## module detailed_router
    ${IRT_MODULE}/detailed_router/DetailedRouter.cpp
)

target_link_libraries(irt_detailed_router
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_detailed_router
    PUBLIC
    ${IRT_MODULE}/detailed_router/dr_data_manager
    ${IRT_MODULE}/detailed_router
)
