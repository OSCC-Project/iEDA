AUX_SOURCE_DIRECTORY(./ TCL_IPW_SRC)

add_library(tcl_ipw
    ${TCL_IPW_SRC}
)

target_link_libraries(tcl_ipw 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
        pwr-cmd
)

target_include_directories(tcl_ipw
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}     
)
