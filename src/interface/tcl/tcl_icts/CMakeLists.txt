AUX_SOURCE_DIRECTORY(./ TCL_ICTS_SRC)

add_library(tcl_icts 
    ${TCL_ICTS_SRC}
)

target_link_libraries(tcl_icts 
    PUBLIC
        tool_manager
        tcl_util
        tcl
        shell-cmd
        str
        icts_api
        tool_api_icts
)

target_include_directories(tcl_icts
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
