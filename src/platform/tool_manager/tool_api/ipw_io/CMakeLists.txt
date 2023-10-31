add_library(tool_api_ipw
    ipw_io.cpp
)

target_include_directories(tool_api_ipw
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_ipw 
    PRIVATE       
    ista-engine
    sta
    power
    flow_config
    idm
    idb
)
