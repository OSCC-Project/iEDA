add_library(eval_api_external_libs INTERFACE)

target_link_libraries(eval_api_external_libs
    INTERFACE
        # platform
        tool_manager
        idm
)

target_include_directories(eval_api_external_libs
    INTERFACE
        # platform
        ${HOME_PLATFORM}
        ${HOME_PLATFORM}/tool_manager
        ${HOME_PLATFORM}/data_manager
)