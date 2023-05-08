AUX_SOURCE_DIRECTORY(./ TCL_CONFIG_SRC)

add_library(tcl_config 
    ${TCL_CONFIG_SRC}
)

target_link_libraries(tcl_config 
    PUBLIC
        tcl
        shell-cmd
        str
        flow
        tool_manager
)

target_include_directories(tcl_config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
