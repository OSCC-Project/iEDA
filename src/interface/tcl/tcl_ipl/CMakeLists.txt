AUX_SOURCE_DIRECTORY(./ TCL_IPL_SRC)

add_library(tcl_ipl 
    ${TCL_IPL_SRC}
)

target_link_libraries(tcl_ipl 
    PUBLIC
        tcl
        shell-cmd
        str
        tool_manager
        ipl-api
        tool_api_ipl
        tcl_util
)

target_include_directories(tcl_ipl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}    
)
