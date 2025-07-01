## set path 
set(IDRC_DATA_MANAGER ${IDRC_SOURCE}/data_manager)
set(IDRC_MODULE ${IDRC_SOURCE}/module)
set(IDRC_TOOLKIT ${IDRC_SOURCE}/toolkit)

## build
add_subdirectory(${IDRC_DATA_MANAGER})
add_subdirectory(${IDRC_MODULE})
add_subdirectory(${IDRC_TOOLKIT})

add_library(idrc_source INTERFACE)

target_link_libraries(idrc_source
    INTERFACE
    idrc_data_manager
    idrc_module
    idrc_toolkit
)
