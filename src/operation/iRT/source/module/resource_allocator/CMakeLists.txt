if(DEBUG_IRT_RESOURCE_ALLOCATOR)
    message(STATUS "RT: DEBUG_IRT_RESOURCE_ALLOCATOR")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_RESOURCE_ALLOCATOR")   
    set(CMAKE_BUILD_TYPE "Release") 
endif()

add_library(irt_resource_allocator
    ## module resource_allocator
    ${IRT_MODULE}/resource_allocator/ResourceAllocator.cpp
)

target_link_libraries(irt_resource_allocator 
    PUBLIC
        irt_data_manager
        irt_module
        irt_solver
        irt_utility
)

target_include_directories(irt_resource_allocator 
    PUBLIC
        ${IRT_MODULE}/resource_allocator/ra_data_manager
        ${IRT_MODULE}/resource_allocator
)
