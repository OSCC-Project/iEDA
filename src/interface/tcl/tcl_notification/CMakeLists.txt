add_library(tcl_notification 
    ${HOME_INTERFACE}/tcl/tcl_notification/src/tcl_init_notification.cpp
)

target_link_libraries(tcl_notification 
    PRIVATE
        tcl_util
        notification
)

target_include_directories(tcl_notification
    PUBLIC
        ${HOME_INTERFACE}/tcl/tcl_notification/include
)
