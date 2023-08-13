# Copyright (c) 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(FindPackageHandleStandardArgs)

# Define search paths based on user input and environment variables
set(TBB_SEARCH_DIR ${TBB_LIBRARY_DIR} ${TBB_ROOT_DIR} ${TBB_DIR} $ENV{TBB_INSTALL_DIR} $ENV{TBBROOT})

# Firstly search for TBB in config mode (i.e. search for TBBConfig.cmake).
find_package(TBB CONFIG HINTS ${TBB_SEARCH_DIR})
if (TBB_FOUND)
    find_package_handle_standard_args(TBB CONFIG_MODE)
    return()
endif()

if (NOT TBB_FIND_COMPONENTS)
    set(TBB_FIND_COMPONENTS tbb tbbmalloc)
    foreach (_tbb_component ${TBB_FIND_COMPONENTS})
        set(TBB_FIND_REQUIRED_${_tbb_component} 1)
    endforeach()
endif()

# for windows add additional default search paths
if(WIN32)
    set(TBB_DEFAULT_SEARCH_DIR  "C:/Program Files/Intel/TBB"
                                "C:/Program Files (x86)/Intel/TBB")

    if (CMAKE_CL_64)
        list(APPEND TBB_LIB_PATH_SUFFIXES lib/intel64/vc14)
        list(APPEND TBB_LIB_PATH_SUFFIXES bin/intel64/vc14)
    else ()
        list(APPEND TBB_LIB_PATH_SUFFIXES lib/ia32/vc14)
        list(APPEND TBB_LIB_PATH_SUFFIXES bin/ia32/vc14)
    endif ()

    list(APPEND ADDITIONAL_LIB_DIRS ENV PATH ENV LIB)
    list(APPEND ADDITIONAL_INCLUDE_DIRS ENV INCLUDE ENV CPATH)
elseif(APPLE)
    foreach (i tbb tbb@2020 tbb@2020_U1 tbb@2020_U2 tbb@2020_U3 tbb@2020_U3_1)
        foreach (j 2020_U1 2020_U2 2020_U3 2020_U3_1)
            list(APPEND TBB_LIB_PATH_SUFFIXES ${i}/${j}/lib)
        endforeach()
    endforeach()
else()
    list(APPEND ADDITIONAL_LIB_DIRS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH)
    list(APPEND ADDITIONAL_INCLUDE_DIRS ENV CPATH ENV C_INCLUDE_PATH ENV CPLUS_INCLUDE_PATH ENV INCLUDE_PATH)
endif()

find_path(_tbb_include_dir
    NAMES oneapi/tbb.h tbb/tbb.h
    HINTS ${TBB_SEARCH_DIR}
    PATHS ${ADDITIONAL_INCLUDE_DIRS} ${TBB_DEFAULT_SEARCH_DIR})

if(_tbb_include_dir)
    foreach (_tbb_version_file "${_tbb_include_dir}/oneapi/tbb/version.h" "${_tbb_include_dir}/tbb/tbb_stddef.h")
        if(EXISTS "${_tbb_version_file}")
            file(READ "${_tbb_version_file}" _tbb_version_info)
            break()
        endif()
    endforeach()
    string(REGEX REPLACE ".*#define TBB_VERSION_MAJOR ([0-9]+).*" "\\1" TBB_VERSION_MAJOR "${_tbb_version_info}")
    string(REGEX REPLACE ".*#define TBB_VERSION_MINOR ([0-9]+).*" "\\1" TBB_VERSION_MINOR "${_tbb_version_info}")
    string(REGEX REPLACE ".*#define TBB_INTERFACE_VERSION ([0-9]+).*" "\\1" TBB_INTERFACE_VERSION "${_tbb_version_info}")
    set(TBB_VERSION "${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}")
    set(_TBB_BUILD_MODES RELEASE DEBUG)
    set(_TBB_DEBUG_SUFFIX _debug)

    if((DEFINED TBB_FIND_VERSION) AND (TBB_VERSION VERSION_GREATER_EQUAL TBB_FIND_VERSION))
        foreach (_tbb_component ${TBB_FIND_COMPONENTS})
            if (NOT TARGET TBB::${_tbb_component})
                add_library(TBB::${_tbb_component} SHARED IMPORTED)
                set_property(TARGET TBB::${_tbb_component} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${_tbb_include_dir})

                foreach(_TBB_BUILD_MODE ${_TBB_BUILD_MODES})
                    set(_tbb_component_lib_name ${_tbb_component}${_TBB_${_TBB_BUILD_MODE}_SUFFIX})

                    if(WIN32)
                        find_library(${_tbb_component_lib_name}_lib
                            NAMES ${_tbb_component_lib_name}12.lib ${_tbb_component_lib_name}
                            HINTS ${TBB_SEARCH_DIR}
                            PATHS ${TBB_DEFAULT_SEARCH_DIR} ${ADDITIONAL_LIB_DIRS}
                            PATH_SUFFIXES ${TBB_LIB_PATH_SUFFIXES})
                        find_file(${_tbb_component_lib_name}_dll
                            NAMES ${_tbb_component_lib_name}.dll
                            HINTS ${TBB_SEARCH_DIR}
                            PATHS ${TBB_DEFAULT_SEARCH_DIR} ${ADDITIONAL_LIB_DIRS}
                            PATH_SUFFIXES ${TBB_LIB_PATH_SUFFIXES})

                        set_target_properties(TBB::${_tbb_component} PROPERTIES
                                              IMPORTED_LOCATION_${_TBB_BUILD_MODE} "${${_tbb_component_lib_name}_dll}"
                                              IMPORTED_IMPLIB_${_TBB_BUILD_MODE}   "${${_tbb_component_lib_name}_lib}"
                                              )
                    elseif(APPLE)
                        find_library(${_tbb_component_lib_name}_so
                            NAMES lib${_tbb_component_lib_name}.so.12 lib${_tbb_component_lib_name}.12.dylib lib${_tbb_component_lib_name}.dylib
                            HINTS ${TBB_SEARCH_DIR}
                            PATHS ${ADDITIONAL_LIB_DIRS} /usr/local/Cellar/
                            PATH_SUFFIXES ${TBB_LIB_PATH_SUFFIXES})

                        set_target_properties(TBB::${_tbb_component} PROPERTIES
                                              IMPORTED_LOCATION_${_TBB_BUILD_MODE} "${${_tbb_component_lib_name}_so}"
                                              )
                    else() # Linux etc.
                        find_library(${_tbb_component_lib_name}_so lib${_tbb_component_lib_name}.so.12 lib${_tbb_component_lib_name}.12.dylib lib${_tbb_component_lib_name}.so.2
                            HINTS ${TBB_SEARCH_DIR}
                            PATHS ${ADDITIONAL_LIB_DIRS} ${TBB_DEFAULT_SEARCH_DIR}
                            PATH_SUFFIXES ${TBB_LIB_PATH_SUFFIXES})

                        set_target_properties(TBB::${_tbb_component} PROPERTIES
                                              IMPORTED_LOCATION_${_TBB_BUILD_MODE} "${${_tbb_component_lib_name}_so}"
                                              )
                    endif()

                    if (${_tbb_component_lib_name}_lib AND ${_tbb_component_lib_name}_dll OR ${_tbb_component_lib_name}_so)
                        set_property(TARGET TBB::${_tbb_component} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_TBB_BUILD_MODE})
                        list(APPEND TBB_IMPORTED_TARGETS TBB::${_tbb_component})
                        set(TBB_${_tbb_component}_FOUND 1)
                    endif()

                    if(MSVC)
                        set(TBB_${_tbb_component_lib_name}_DLL ${${_tbb_component_lib_name}_dll})
                    endif()
                    unset(${_tbb_component_lib_name}_lib CACHE)
                    unset(${_tbb_component_lib_name}_dll CACHE)
                    unset(${_tbb_component_lib_name}_so CACHE)
                    unset(_tbb_component_lib_name)
                endforeach()
            endif()
        endforeach()
        unset(_TBB_BUILD_MODESS)
        unset(_TBB_DEBUG_SUFFIX)
    endif()
endif()
unset(_tbb_include_dir CACHE)

list(REMOVE_DUPLICATES TBB_IMPORTED_TARGETS)

find_package_handle_standard_args(TBB
    REQUIRED_VARS TBB_IMPORTED_TARGETS
    VERSION_VAR TBB_VERSION
    HANDLE_COMPONENTS)
