add_library(idrc_api_external_lib INTERFACE)

target_link_libraries(idrc_api_external_lib 
    INTERFACE
        idm
)

target_include_directories(idrc_api_external_lib
    INTERFACE
        ${HOME_PLATFORM}/data_manager
        ${HOME_PLATFORM}/data_manager/file_manager
)