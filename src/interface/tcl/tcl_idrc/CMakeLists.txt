add_library(tcl_idrc 
    ${HOME_INTERFACE}/tcl/tcl_idrc/src/tcl_check_def.cpp
    ${HOME_INTERFACE}/tcl/tcl_idrc/src/tcl_init_drc.cpp
    ${HOME_INTERFACE}/tcl/tcl_idrc/src/tcl_destroy_drc.cpp
    ${HOME_INTERFACE}/tcl/tcl_idrc/src/tcl_drc_cmp_violation.cpp
)

target_link_libraries(tcl_idrc 
    PUBLIC
        tool_manager
        tcl_util
    PRIVATE
        idrc_interface
)

target_include_directories(tcl_idrc
    PUBLIC
        ${HOME_INTERFACE}/tcl/tcl_idrc/include
)