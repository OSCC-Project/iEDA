AUX_SOURCE_DIRECTORY(./ TCL_LM_SRC)

add_library(tcl_lm
    ${TCL_LM_SRC}
)

target_link_libraries(tcl_lm 
    PUBLIC
        tcl
        shell-cmd
        ilm_api
)

target_include_directories(tcl_lm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
