AUX_SOURCE_DIRECTORY(./ TCL_CONTEST_SRC)

add_library(tcl_contest
    ${TCL_CONTEST_SRC}
)

target_link_libraries(tcl_contest 
    PUBLIC
        tcl
        shell-cmd
        contest_flow
)

target_include_directories(tcl_contest
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
