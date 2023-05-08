if(BUILD_GUI)
    add_subdirectory(gui)
endif()

add_subdirectory(tcl)
add_subdirectory(shell)

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()
