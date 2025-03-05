# cmake version requirements
cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_BUILD_TYPE "Debug")

option(BUILD_QT "If ON, build QT." OFF)
if(BUILD_QT)
    add_definitions(-DBUILD_QT)
    message("-- iPL: ENABLE QT FOR PLOT")
endif()

# set
set(HOME_iPL ${HOME_OPERATION}/iPL)
set(iPL_API ${HOME_iPL}/api)
set(iPL_EXTERNAL_LIBS ${HOME_iPL}/external_libs)
set(iPL_SOURCE ${HOME_iPL}/source)
set(iPL_TEST ${HOME_iPL}/test)

# sub directory
add_subdirectory(${iPL_API})
add_subdirectory(${iPL_EXTERNAL_LIBS})
add_subdirectory(${iPL_SOURCE})
add_subdirectory(${iPL_TEST})
