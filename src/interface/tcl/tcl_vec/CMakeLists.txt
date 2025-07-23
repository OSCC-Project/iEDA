AUX_SOURCE_DIRECTORY(./ TCL_VEC_SRC)

add_library(tcl_vec
    ${TCL_VEC_SRC}
)

target_link_libraries(tcl_vec 
    PUBLIC
        tcl
        shell-cmd
        ivec_api
)

target_include_directories(tcl_vec
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
