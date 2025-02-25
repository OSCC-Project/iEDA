AUX_SOURCE_DIRECTORY(./ TCL_IDRC_SRC)

add_library(tcl_idrc 
    ${TCL_IDRC_SRC}
)

target_link_libraries(tcl_idrc 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
        tcl_util
        # idrc_api
        tool_api_idrc
        idrc_pro_api
        ${LIBPROFILER_LIBRARY}
)

target_include_directories(tcl_idrc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
