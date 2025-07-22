
add_subdirectory(config)

add_library(flow
  flow.cpp
)

target_link_libraries(flow tool_manager flow_config ieda_tcl file_manager_placement file_manager_cts 
# file_manager_drc
)
target_include_directories(flow 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/config

        ${HOME_PLATFORM}/tool_manager
        ${HOME_PLATFORM}/tool_manager/module
)




