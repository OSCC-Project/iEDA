cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

find_package(TCL 8.6 QUIET)
if(TCL_FOUND)
  find_path(TCL_INCLUDE_DIR tcl.h PATH_SUFFIXES tcl8.6)
  include_directories(${TCL_INCLUDE_PATH})
else(TCL_FOUND)
  # install Tcl 8.6 https://github.com/tcltk/tcl/tree/core-8-6-branch
  message(FATAL_ERROR "Could not find TCL!")
endif(TCL_FOUND)

aux_source_directory(./ SRC)
set(LINK_tcl "tcl8.6")
find_library(TCL_STATIC_LIB 
  NAMES libtcl8.6.a
  PATHS /usr/lib/x86_64-linux-gnu
  NO_DEFAULT_PATH
)

if(BUILD_STATIC_LIB AND TCL_STATIC_LIB)
  set(LINK_tcl ${TCL_STATIC_LIB})
  add_library(tcl STATIC ${SRC})
else()
  add_library(tcl SHARED ${SRC})
endif()

target_link_libraries(tcl PUBLIC ${LINK_tcl} absl::inlined_vector)

target_include_directories(tcl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
