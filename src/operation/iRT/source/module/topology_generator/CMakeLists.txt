if(DEBUG_IRT_TOPOLOGY_GENERATOR)
    message(STATUS "RT: DEBUG_IRT_TOPOLOGY_GENERATOR")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_TOPOLOGY_GENERATOR")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_topology_generator
    ## module topology_generator
    ${IRT_MODULE}/topology_generator/TopologyGenerator.cpp
)

target_link_libraries(irt_topology_generator
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_topology_generator
    PUBLIC
    ${IRT_MODULE}/topology_generator/tg_data_manager
    ${IRT_MODULE}/topology_generator

)
