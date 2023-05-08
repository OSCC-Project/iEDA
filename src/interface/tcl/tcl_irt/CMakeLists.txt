add_library(tcl_irt 
    ${HOME_INTERFACE}/tcl/tcl_irt/egr/tcl_run_egr.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/rt/tcl_destroy_rt.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/rt/tcl_init_rt.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/rt/tcl_run_rt.cpp
)

target_link_libraries(tcl_irt 
    PUBLIC
        tool_manager
        tcl_util
    PRIVATE
        irt_api
)

target_include_directories(tcl_irt
    PUBLIC
        ${HOME_INTERFACE}/tcl/tcl_irt/egr
        ${HOME_INTERFACE}/tcl/tcl_irt/rt
        ${HOME_INTERFACE}/tcl/tcl_irt
)