AUX_SOURCE_DIRECTORY(./ TCL_INO_SRC)

add_library(tcl_ino 
    ${TCL_INO_SRC}
)

target_link_libraries(tcl_ino 
    PUBLIC
        tcl_util
        tool_manager
        tcl
        shell-cmd
        str
)

target_include_directories(tcl_ino
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}       
)
