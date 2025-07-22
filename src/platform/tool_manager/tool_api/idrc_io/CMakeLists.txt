add_library(tool_api_idrc
    idrc_io.cpp
)

target_include_directories(tool_api_idrc
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)
if(USE_PROFILER)
    add_definitions(-DUSE_PROFILER)
endif(USE_PROFILER)

target_link_libraries(tool_api_idrc
    PUBLIC 
    idrc_interface
    PRIVATE
        flow_config
        file_manager_drc 
        ieda_report
        idm
        ieda_feature
        
)
