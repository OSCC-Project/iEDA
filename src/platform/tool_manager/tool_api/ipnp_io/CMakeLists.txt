add_library(tool_api_ipnp
    ipnp_io.cpp
)

target_include_directories(tool_api_ipnp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                        
)

target_link_libraries(tool_api_ipnp
    PRIVATE
        flow_config
        ipnp_api
)
