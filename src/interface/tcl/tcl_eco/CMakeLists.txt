AUX_SOURCE_DIRECTORY(./ TCL_ECO_SRC)

add_library(tcl_eco
    ${TCL_ECO_SRC}
)

target_link_libraries(tcl_eco 
    PUBLIC
        tcl
        shell-cmd
        ieco_api
        idm
        idb
)

target_include_directories(tcl_eco
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
