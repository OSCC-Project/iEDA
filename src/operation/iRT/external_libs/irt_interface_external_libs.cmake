add_library(irt_interface_external_libs INTERFACE)

target_link_libraries(irt_interface_external_libs
    INTERFACE
        eval_source
        tool_manager
        idm
        idrc_pro_api
        feature_db
)

target_include_directories(irt_interface_external_libs
    INTERFACE
        ${HOME_PLATFORM}/tool_manager
        ${HOME_PLATFORM}/tool_manager/tool_api
        ${HOME_PLATFORM}/tool_manager/tool_api/irt_io
        ${HOME_PLATFORM}/data_manager
        ${HOME_PLATFORM}/data_manager/file_manager
        ${HOME_FEATURE}/database
)
