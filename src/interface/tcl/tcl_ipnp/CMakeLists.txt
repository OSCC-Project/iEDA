AUX_SOURCE_DIRECTORY(./ TCL_IPNP_SRC)

add_library(tcl_ipnp 
    ${TCL_IPNP_SRC}
)

target_link_libraries(tcl_ipnp 
    PUBLIC 
        tool_manager
        tcl 
        shell-cmd 
        str
        pnp
        iPNPApi
)

target_include_directories(tcl_ipnp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
) 