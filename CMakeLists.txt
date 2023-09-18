# ***************************************************************************************
# Copyright (c) 2023-2025 Peng Cheng Laboratory
# Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
# Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
#
# iEDA is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
# ***************************************************************************************

cmake_minimum_required(VERSION 3.11)
project(iEDA)

# option setting
option(BUILD_PR_FLOW "If ON, build PR flow." ON)
option(BUILD_STATIC_LIB "If ON, build static lib." ON)

if(NOT BUILD_STATIC_LIB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

option(BUILD_PYTHON "If ON, build python interface." OFF)
if(BUILD_PYTHON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
option(BUILD_GUI "If ON, build GUI." OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(FORT_ENABLE_TESTING OFF CACHE INTERNAL "") # Disable building tests and examples in libfort project

# SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
#set(BUILD_SHARED_LIBS ON)

if(NOT DEFINED CMD_BUILD)
    set(SANITIZER OFF)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endif()

if(SANITIZER)
    add_compile_options("-fsanitize=address")
    link_libraries("-fsanitize=address")
endif()

include(cmake/setting.cmake)
include(cmake/third_party.cmake)
include(cmake/operation/ista.cmake)
include(cmake/operation/ipw.cmake)

if(BUILD_GUI)
    add_definitions(-DBUILD_GUI)
endif()

option(CONTEST "If ON, run contest porject." ON)
if(CONTEST)
    add_definitions(-DCONTEST)
    add_subdirectory(contest)
endif()

add_subdirectory(src/third_party)
add_subdirectory(src/utility)
add_subdirectory(src/apps)
add_subdirectory(src/database)
add_subdirectory(src/interface)
add_subdirectory(src/evaluation)
add_subdirectory(src/analysis)
add_subdirectory(src/operation)
add_subdirectory(src/platform)
add_subdirectory(src/solver)