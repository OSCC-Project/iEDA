message(STATUS "MP: Configuring")

set(CMAKE_CXX_STANDARD 20)

set(HOME_IMP ${HOME_OPERATION}/iMP)
set(IMP_API ${HOME_IMP}/api)
set(IMP_EXTERNAL_LIBS ${HOME_IMP}/external_libs)
set(IMP_SOURCE ${HOME_IMP}/source)
set(IMP_TEST ${HOME_IMP}/test)
set(IMP_OPS ${HOME_IMP}/ops)

add_subdirectory(${IMP_API})
add_subdirectory(${IMP_EXTERNAL_LIBS})
add_subdirectory(${IMP_SOURCE})
add_subdirectory(${IMP_TEST})
add_subdirectory(${IMP_OPS})

message(STATUS "MP: Done")

