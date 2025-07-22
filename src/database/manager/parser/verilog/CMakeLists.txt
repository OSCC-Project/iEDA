cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Release")

if(BUILD_STATIC_LIB)
  # This is a workaround on cmake 3.22.1, should ensure static libraries are used
  # Note: There is a better approach in newer versions of CMake:
  # set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "Use static zlib" FORCE)

  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

  find_package(ZLIB REQUIRED)

  set(CMAKE_FIND_LIBRARY_SUFFIXES ${ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

  if(ZLIB_LIBRARIES MATCHES "\\.so(\\.|$)")
    message(FATAL_ERROR "Found dynamic zlib, but static required: ${ZLIB_LIBRARIES}")
  endif()

  message(STATUS "Linking static zlib: ${ZLIB_LIBRARIES}")
else()
  find_package(ZLIB REQUIRED)
endif()

aux_source_directory(./ SRC)

add_library(verilog ${SRC})

set(RUST_PROJECT_NAME verilog_parser)
set(RUST_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/verilog-rust/verilog-parser)
set(RUST_LIB_TYPE a)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/debug/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "")
else()
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/release/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "--release")
endif()


message(STATUS "verilog parser rust lib path ${RUST_LIB_PATH}")

ADD_EXTERNAL_PROJ(verilog)

target_link_libraries(verilog str log ${ZLIB_LIBRARIES} ${RUST_LIB_PATH} dl) 

target_include_directories(verilog
    PUBLIC
        ${HOME_OPERATION}/iSTA/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)