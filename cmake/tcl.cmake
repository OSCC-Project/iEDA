find_package(TCL 8.6 QUIET)
IF(TCL_FOUND)
    INCLUDE_DIRECTORIES(${TCL_INCLUDE_PATH})
ELSE(TCL_FOUND)
    # install Tcl 8.6 https://github.com/tcltk/tcl/tree/core-8-6-branch
    MESSAGE(FATAL_ERROR "Could not find TCL!")
ENDIF(TCL_FOUND)

include_directories(${HOME_UTILITY}/tcl)