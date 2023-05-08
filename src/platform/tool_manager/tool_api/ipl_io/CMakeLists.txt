add_library(tool_api_ipl
    ipl_io.cpp
)

target_include_directories(tool_api_ipl
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_ipl
    PUBLIC
        flow_config
        file_manager_placement
        file_manager_base
        ipl-api   
        idm
)