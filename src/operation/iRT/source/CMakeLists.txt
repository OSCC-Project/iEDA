## set path 
set(IRT_DATA_MANAGER ${IRT_SOURCE}/data_manager)
set(IRT_MODULE ${IRT_SOURCE}/module)
set(IRT_TOOLKIT ${IRT_SOURCE}/toolkit)

## build
add_subdirectory(${IRT_DATA_MANAGER})
add_subdirectory(${IRT_MODULE})
add_subdirectory(${IRT_TOOLKIT})

add_library(irt_source INTERFACE)

target_link_libraries(irt_source
    INTERFACE
    irt_data_manager
    irt_module
    irt_toolkit
)
