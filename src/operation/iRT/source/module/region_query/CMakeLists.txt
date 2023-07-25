if(DEBUG_IRT_REGION_QUERY)
    message(STATUS "RT: DEBUG_IRT_REGION_QUERY")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: DEBUG_IRT_REGION_QUERY")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(irt_region_query
    ## module region_query
    ${IRT_MODULE}/region_query/RegionQuery.cpp
)

target_link_libraries(irt_region_query 
    PUBLIC
        irt_data_manager
        irt_module
        irt_solver
        irt_utility
)

target_include_directories(irt_region_query 
    PUBLIC
        ${IRT_MODULE}/region_query/rq_data_manager
        ${IRT_MODULE}/region_query
)
