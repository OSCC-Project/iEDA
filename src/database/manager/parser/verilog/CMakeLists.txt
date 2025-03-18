cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Release")

find_package(ZLIB REQUIRED)

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