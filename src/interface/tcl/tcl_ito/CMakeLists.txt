AUX_SOURCE_DIRECTORY(./ TCL_ITO_SRC)

add_library(tcl_ito 
    ${TCL_ITO_SRC}
)

target_link_libraries(tcl_ito 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
        tcl_util
        ito_api
)

target_include_directories(tcl_ito
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
       
)
