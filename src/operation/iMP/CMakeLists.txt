message(STATUS "MP: Configuring")

set(CMAKE_CXX_STANDARD 20)

set(HOME_IMP ${HOME_OPERATION}/iMP)
set(IMP_API ${HOME_IMP}/api)
set(IMP_EXTERNAL_LIBS ${HOME_IMP}/external_libs)
set(IMP_SOURCE ${HOME_IMP}/source)
set(IMP_TEST ${HOME_IMP}/test)
set(IMP_OPS ${HOME_IMP}/ops)

message(STATUS "MP: Python executable - " ${PYTHON_EXECUTABLE})
add_definitions("-DPYTHON=\"${PYTHON_EXECUTABLE}\"")
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import matplotlib"
    RESULT_VARIABLE Matplotlib_NOT_FOUND
    OUTPUT_QUIET
)
if (Matplotlib_NOT_FOUND)
    message(STATUS "MP: matplotlib library not found.")
else()
    message(STATUS "MP: matplotlib library found.")
    add_definitions(-DBUILD_PLOT)
    execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import imageio"
    RESULT_VARIABLE imageio_NOT_FOUND
    OUTPUT_QUIET
    )
    if (imageio_NOT_FOUND)
        message(STATUS "MP: imageio library not found.")
    else()
        message(STATUS "MP: imageio library found.")
        add_definitions(-DBUILD_GIF)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
add_definitions(-DRELEASE_MOD)
endif()

add_subdirectory(${IMP_API})
add_subdirectory(${IMP_EXTERNAL_LIBS})
add_subdirectory(${IMP_SOURCE})
add_subdirectory(${IMP_TEST})

message(STATUS "MP: Done")

