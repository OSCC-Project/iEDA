add_library(tool_api_idrc
    idrc_io.cpp
)

target_include_directories(tool_api_idrc
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_idrc
    PUBLIC 
        idrc_api
    PRIVATE
        flow_config
        idrc_db
        file_manager_drc 
        ieda_report
        idm
)