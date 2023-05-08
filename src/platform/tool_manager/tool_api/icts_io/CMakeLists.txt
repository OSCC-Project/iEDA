add_library(tool_api_icts
    icts_io.cpp
)

target_include_directories(tool_api_icts
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_icts
    PRIVATE
        flow_config
        file_manager_cts
        ista-engine
        idm
        sta
)