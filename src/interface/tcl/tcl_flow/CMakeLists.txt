AUX_SOURCE_DIRECTORY(./ TCL_FLOW_SRC)

add_library(tcl_flow 
    ${TCL_FLOW_SRC}
)

target_link_libraries(tcl_flow 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
        flow
        tcl_util
)

target_include_directories(tcl_flow
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
