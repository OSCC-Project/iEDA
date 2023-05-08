# header file
include_directories(${iPL_SOURCE})

# set
set(iPL_CONFIG ${iPL_SOURCE}/config)
set(iPL_MODULE ${iPL_SOURCE}/module)
set(iPL_SOLVER ${iPL_SOURCE}/solver)
set(iPL_UTILITY ${iPL_SOURCE}/utility)

# sub directory
add_subdirectory(${iPL_CONFIG})
add_subdirectory(${iPL_MODULE})
add_subdirectory(${iPL_SOLVER})
add_subdirectory(${iPL_UTILITY})

# build library
add_library(ipl-source
        PlacerDB.cc
)

# link library
target_link_libraries(ipl-source 
    PUBLIC
        ipl-module-grid_manager
        ipl-module-topology_manager
        ipl-module-wrapper
        ipl-module-logger
        ipl-configurator
        ipl-module
        ipl-solver
        ipl-utility
)

target_include_directories(ipl-source PUBLIC ${iPL_SOURCE})