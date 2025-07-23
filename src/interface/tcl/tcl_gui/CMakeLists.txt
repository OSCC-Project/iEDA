AUX_SOURCE_DIRECTORY(./ TCL_GUI_SRC)

add_library(tcl_gui 
    ${TCL_GUI_SRC}
)

target_link_libraries(tcl_gui 
    PUBLIC
        tool_manager
        tcl
        shell-cmd
        str
        iGUI
        tcl_qt
        ivec_api
)

target_include_directories(tcl_gui
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_INTERFACE}/gui/interface
        
)
