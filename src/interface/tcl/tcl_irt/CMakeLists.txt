add_library(tcl_irt 
    ${HOME_INTERFACE}/tcl/tcl_irt/src/tcl_clear_def.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/src/tcl_destroy_rt.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/src/tcl_init_rt.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/src/tcl_output_db_json.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/src/tcl_run_egr.cpp
    ${HOME_INTERFACE}/tcl/tcl_irt/src/tcl_run_rt.cpp
)

target_link_libraries(tcl_irt 
    PUBLIC
        tool_manager
        tcl_util
    PRIVATE
        irt_interface
)

target_include_directories(tcl_irt
    PUBLIC
        ${HOME_INTERFACE}/tcl/tcl_irt/include
)