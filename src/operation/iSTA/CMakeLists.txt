cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)
# set(CMAKE_BUILD_TYPE "Debug")
#include(cmake/conanbuildinfo.cmake)
#conan_basic_setup()

if(TEST_SHELL)
    add_definitions(-DTEST_SHELL)
endif(TEST_SHELL)

# ADD_COMPILE_OPTIONS("-fsanitize=address")
# LINK_LIBRARIES("-fsanitize=address")

include_directories(${HOME_UTILITY}/stdBase)
include_directories(${HOME_UTILITY}/stdBase/include)
include_directories(${HOME_UTILITY})
include_directories(${HOME_DATABASE}/manager/parser)
include_directories(SYSTEM ${HOME_THIRDPARTY})
include_directories(${HOME_OPERATION}/iSTA)
include_directories(${HOME_OPERATION}/iSTA/source/module)
include_directories(${HOME_OPERATION}/iSTA/source/module/include)

include(${HOME_CMAKE}/operation/idb.cmake)

link_directories(${CMAKE_BINARY_DIR}/lib)

find_package(Eigen3 REQUIRED)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 ${EIGEN3_INCLUDE_DIR}")
link_directories(${EIGEN3_LIBRARY})

option(UPDATE_RELEASE_VERSION "update git version for release" OFF)

if (UPDATE_RELEASE_VERSION)
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=40 --always OUTPUT_VARIABLE GIT_VERSION)
        string(STRIP "${GIT_VERSION}" GIT_VERSION)
        message(STATUS "Git version: " ${GIT_VERSION})
    endif()

    configure_file(
        "${HOME_OPERATION}/iSTA/source/module/include/Version.hh.in"
        "${HOME_OPERATION}/iSTA/source/module/include/Version.hh"
    )
endif()


add_subdirectory(api)
add_subdirectory(source/module/netlist)
add_subdirectory(source/module/sdc)
add_subdirectory(source/module/sdc-cmd)
add_subdirectory(source/module/shell-cmd)
add_subdirectory(source/module/sta)
add_subdirectory(source/module/delay)
add_subdirectory(source/solver)
add_subdirectory(source/utility)

add_subdirectory(test)

add_executable(iSTA main.cc)

target_link_libraries(iSTA sdc ista-engine
sdc-cmd shell-cmd sta log str time netlist liberty
delay utility sta-solver verilog-parser graph idb
tcl usage pthread stdc++fs IdbBuilder def_service lef_service)