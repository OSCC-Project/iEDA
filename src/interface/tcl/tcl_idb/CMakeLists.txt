AUX_SOURCE_DIRECTORY(./ TCL_IDB_SRC)

add_library(tcl_idb 
    ${TCL_IDB_SRC}
)

target_link_libraries(tcl_idb 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
        idm
        ieda_report
)

target_include_directories(tcl_idb
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
