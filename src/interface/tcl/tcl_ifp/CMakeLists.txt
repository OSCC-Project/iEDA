AUX_SOURCE_DIRECTORY(./ TCL_IFP_SRC)

add_library(tcl_ifp 
    ${TCL_IFP_SRC}
)

target_link_libraries(tcl_ifp 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
        ifp_api
)

target_include_directories(tcl_ifp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  
)
