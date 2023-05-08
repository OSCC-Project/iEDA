AUX_SOURCE_DIRECTORY(./ TCL_UTIL_SRC)

add_library(tcl_util 
    ${TCL_UTIL_SRC}
)

target_link_libraries(tcl_util 
    PUBLIC
        tcl
        shell-cmd
)

target_include_directories(tcl_util
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
