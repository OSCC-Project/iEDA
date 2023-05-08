add_library(flow_config
        flow_config.cpp
)

target_link_libraries(flow_config str)

target_include_directories(flow_config 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/string
    ${HOME_UTILITY}/json
    ${HOME_THIRDPARTY}/json
)