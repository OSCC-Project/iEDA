if(DEBUG_IRT_GDS_PLOTTER)
    message(STATUS "RT: DEBUG_IRT_GDS_PLOTTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_GDS_PLOTTER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_gds_plotter
    ## module gds_plotter
    ${IRT_MODULE}/gds_plotter/GDSPlotter.cpp
)

target_link_libraries(irt_gds_plotter
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_gds_plotter
    PUBLIC
    ${IRT_MODULE}/gds_plotter/gp_data_manager
    ${IRT_MODULE}/gds_plotter
)
