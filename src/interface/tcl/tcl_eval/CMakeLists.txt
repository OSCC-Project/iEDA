include_directories(
    ${CMAKE_SOURCE_DIR}/src/evaluation/src/module/eval_io
    ${CMAKE_SOURCE_DIR}/src/evaluation/src/module/timing
)

AUX_SOURCE_DIRECTORY(./ TCL_EVAL_SRC)

add_library(tcl_eval STATIC
    ${TCL_EVAL_SRC}
)

target_link_libraries(tcl_eval 
    PUBLIC
        tcl
        shell-cmd
        eval_wirelength_api
        eval_congestion_api
        eval_density_api
        eval_timing_api
        eval_io
        idm
        idb
)

target_include_directories(tcl_eval
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
