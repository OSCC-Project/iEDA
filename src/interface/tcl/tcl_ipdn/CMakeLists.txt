AUX_SOURCE_DIRECTORY(./ TCL_IPDN_SRC)

add_library(tcl_ipdn 
    ${TCL_IPDN_SRC}
)

target_link_libraries(tcl_ipdn 
    PUBLIC
        tcl
        shell-cmd
        str
        tool_manager
        ipdn_api
)

target_include_directories(tcl_ipdn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  
)
