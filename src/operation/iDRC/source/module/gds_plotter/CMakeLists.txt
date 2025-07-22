if(DEBUG_IDRC_GDS_PLOTTER)
    message(STATUS "DRC: DEBUG_IDRC_GDS_PLOTTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_GDS_PLOTTER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_gds_plotter
    ## module gds_plotter
    ${IDRC_MODULE}/gds_plotter/GDSPlotter.cpp
)

target_link_libraries(idrc_gds_plotter
    PUBLIC
    idrc_data_manager
    idrc_module
    idrc_toolkit
)

target_include_directories(idrc_gds_plotter
    PUBLIC
    ${IDRC_MODULE}/gds_plotter/gp_data_manager
    ${IDRC_MODULE}/gds_plotter
)
