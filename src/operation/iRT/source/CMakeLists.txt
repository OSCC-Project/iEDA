## set path 
set(IRT_DATA_MANAGER ${IRT_SOURCE}/data_manager)
set(IRT_MODULE ${IRT_SOURCE}/module)
set(IRT_SOLVER ${IRT_SOURCE}/solver)
set(IRT_UTILITY ${IRT_SOURCE}/utility)

## build
add_subdirectory(${IRT_DATA_MANAGER})
add_subdirectory(${IRT_MODULE})
add_subdirectory(${IRT_SOLVER})
add_subdirectory(${IRT_UTILITY})

add_library(irt_source INTERFACE)

target_link_libraries(irt_source
    INTERFACE
        irt_data_manager
        irt_module
        irt_utility
)
