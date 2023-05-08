add_library(tool_api_ista
    ista_io.cpp
)

target_include_directories(tool_api_ista
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_ista 
    PRIVATE       
    ista-engine
    sta
    flow_config
    idm
    idb
)
