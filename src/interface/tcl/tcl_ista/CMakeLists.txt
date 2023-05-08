AUX_SOURCE_DIRECTORY(./ TCL_ISTA_SRC)

add_library(tcl_ista
    ${TCL_ISTA_SRC}
)

target_link_libraries(tcl_ista 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
)

target_include_directories(tcl_ista
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}     
)
