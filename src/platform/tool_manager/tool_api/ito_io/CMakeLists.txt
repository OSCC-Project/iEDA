add_library(tool_api_ito
    ito_io.cpp
)

target_include_directories(tool_api_ito
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_ito
    PRIVATE
        flow_config
        idm
        ito_api
)