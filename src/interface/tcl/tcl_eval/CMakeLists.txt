AUX_SOURCE_DIRECTORY(./ TCL_EVAL_SRC)

add_library(tcl_eval
    ${TCL_EVAL_SRC}
)

target_link_libraries(tcl_eval 
    PUBLIC
        tcl
        shell-cmd
        eval_api
        idm
        idb
)

target_include_directories(tcl_eval
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
