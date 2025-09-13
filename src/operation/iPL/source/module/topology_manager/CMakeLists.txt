add_library(ipl-module-topology_manager TopologyManager.cc)

target_link_libraries(ipl-module-topology_manager 
    PUBLIC 
        ipl-module-logger
)

target_include_directories(ipl-module-topology_manager 
PUBLIC 
    ${iPL_TOPOLOGY_MANAGER}
    ${iPL_SOURCE}
)