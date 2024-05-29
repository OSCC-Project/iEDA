add_library(ito_api_external_libs INTERFACE)

target_link_libraries(ito_api_external_libs 
    INTERFACE
        tool_manager
        idm
        ista-engine
        idb
        IdbBuilder
        feature_db
)

target_include_directories(ito_api_external_libs
    INTERFACE
        ${HOME_PLATFORM}/tool_manager
        ${HOME_PLATFORM}/tool_manager/tool_api
        ${HOME_PLATFORM}/tool_manager/tool_api/ito_io
        ${HOME_PLATFORM}/data_manager
        ${HOME_PLATFORM}/data_manager/file_manager
        ${HOME_FEATURE}/database
)
