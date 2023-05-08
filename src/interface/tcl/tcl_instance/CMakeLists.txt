AUX_SOURCE_DIRECTORY(./ TCL_INST_SRC)

add_library(tcl_inst 
    ${TCL_INST_SRC}
)

target_link_libraries(tcl_inst 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
)

target_include_directories(tcl_inst
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
