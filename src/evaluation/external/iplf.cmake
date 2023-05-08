add_library(eval_iplf_libs INTERFACE)

target_link_libraries(eval_iplf_libs
    INTERFACE
        tool_manager
        idm
)

target_include_directories(eval_iplf_libs
    INTERFACE
        ${HOME_PLATFORM}
        ${HOME_PLATFORM}/tool_manager
        ${HOME_PLATFORM}/data_manager
)
