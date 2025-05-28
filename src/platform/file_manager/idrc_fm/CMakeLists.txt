add_library(file_manager_drc
    file_drc.cpp
)

target_link_libraries(file_manager_drc 
file_manager_base  
idm
tool_api_idrc
)

target_include_directories(file_manager_drc
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
    ${HOME_DATABASE}/interaction/RT_DRC
)