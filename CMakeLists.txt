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
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -Werror=return-type")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -Werror=return-type")
set(CMAKE_BUILD_TYPE "Release")

if(DEFINED BUILD_AIEDA)
    option(BUILD_STATIC_LIB "Build static library OFF" OFF)
    option(BUILD_PYTHON "Build Python bindings ON" ON)
else()
    option(BUILD_STATIC_LIB "Build static library (default ON)" ON)
    option(BUILD_PYTHON "Build Python bindings (default OFF)" OFF)
endif()
option(USE_PROFILER "Enable performance profiling (default OFF)" OFF)
option(SANITIZER "Enable address sanitizer (default OFF)" OFF)
option(BUILD_GUI "Enable GUI components (default OFF)" OFF)
option(USE_GPU "Enable GPU acceleration (default OFF)" OFF)
option(COMPATIBILITY_MODE "Enable compatibility mode (disable aggressive optimizations)" ON)

# Define GLOG_USE_GLOG_EXPORT for glog 0.7.1+ compatibility
add_definitions(-DGLOG_USE_GLOG_EXPORT)

if(NOT DEFINED CMD_BUILD)
    set(SANITIZER OFF)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endif()

if(BUILD_PYTHON AND BUILD_STATIC_LIB)
    message(FATAL_ERROR "Cannot enable both BUILD_PYTHON and BUILD_STATIC_LIB")
endif()

if(SANITIZER AND NOT COMPATIBILITY_MODE)
    message(WARNING "Address sanitizer may conflict with aggressive optimization")
endif()

if(BUILD_PYTHON)
    set(BUILD_STATIC_LIB OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

if (BUILD_STATIC_LIB)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared libs" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
endif()

if(NOT COMPATIBILITY_MODE)
    set(CMAKE_BUILD_TYPE "Release")
    add_compile_options(-flto=auto)
    add_link_options(-flto=auto)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native -fdata-sections -ffunction-sections")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wl,--gc-sections")

    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    set(CMAKE_AR "gcc-ar")
    set(CMAKE_RANLIB "gcc-ranlib")
    
    # diable use gpu for lto confilct with cuda
    set(USE_GPU OFF)
endif()

if(USE_PROFILER)
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    find_package(PROFILER REQUIRED) 
    message("LIBPROFILER_LIBRARY: ${LIBPROFILER_LIBRARY}")
endif()

if(SANITIZER)
    add_definitions(-DUSE_CPP_STD)
    add_compile_options("-fsanitize=address")
    link_libraries("-fsanitize=address")
    message("address sanitizer load")
endif()

if(BUILD_GUI)
    add_definitions(-DBUILD_GUI)
endif()


if (USE_GPU)
    add_definitions(-DUSE_GPU)
    include(cmake/cuda.cmake)
endif()

include(cmake/setting.cmake)
include(cmake/third_party.cmake)
include(cmake/operation/ista.cmake)
include(cmake/operation/ipa.cmake)
include(cmake/rust.cmake)

add_subdirectory(src/third_party)
add_subdirectory(src/utility)
add_subdirectory(src/apps)
add_subdirectory(src/database)
add_subdirectory(src/interface)
add_subdirectory(src/vectorization)
add_subdirectory(src/evaluation)
add_subdirectory(src/operation)
add_subdirectory(src/platform)
add_subdirectory(src/solver)
add_subdirectory(src/feature)