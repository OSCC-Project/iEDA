cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Release")

################################################################
#
# Flex/bison scanner/parsers
#
################################################################
find_package(ZLIB REQUIRED)

# find_package(FLEX)
# find_package(BISON)

# Verilog scan/parse. 
# flex_target(VerilogLex ${HOME_THIRDPARTY}/parser/verilog/mVerilogLex.l ${HOME_THIRDPARTY}/parser/verilog/mVerilogLex.cc)
# bison_target(VerilogParser ${HOME_THIRDPARTY}/parser/verilog/mVerilogParse.y ${HOME_THIRDPARTY}/parser/verilog/mVerilogParse.cc 
# COMPILE_FLAGS "-v -t")
# add_flex_bison_dependency(VerilogLex VerilogParser)

aux_source_directory(./ SRC)

# add_library(verilog-parser ${SRC} ${FLEX_VerilogLex_OUTPUTS} ${BISON_VerilogParser_OUTPUTS})

add_library(verilog ${SRC} 
${HOME_THIRDPARTY}/parser/verilog/mVerilogParse.cc 
${HOME_THIRDPARTY}/parser/verilog/mVerilogLex.cc)

include(ExternalProject)

set(RUST_PROJECT_NAME verilog_parser)
set(RUST_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/verilog-rust/verilog-parser)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/debug/lib${RUST_PROJECT_NAME}.so)
    set(RUST_BUILD_CMD_OPTION "")
else()
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/release/lib${RUST_PROJECT_NAME}.so)
    set(RUST_BUILD_CMD_OPTION "--release")
endif()


message("verilog parser rust lib path ${RUST_LIB_PATH}")

include(ExternalProject)

ExternalProject_Add(
    ${RUST_PROJECT_NAME}
    PREFIX ${RUST_PROJECT_DIR}
    SOURCE_DIR ${RUST_PROJECT_DIR}
    BINARY_DIR ${RUST_PROJECT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cargo build ${RUST_BUILD_CMD_OPTION}
    INSTALL_COMMAND ""
    BUILD_ALWAYS 1
    BUILD_BYPRODUCTS ${RUST_LIB_PATH}
)

add_dependencies(verilog ${RUST_PROJECT_NAME})

target_link_libraries(verilog str log ${ZLIB_LIBRARIES} ${RUST_LIB_PATH}) 

target_include_directories(verilog
    PUBLIC
        ${HOME_OPERATION}/iSTA/include   
        ${HOME_THIRDPARTY}/parser/verilog
        ${CMAKE_CURRENT_SOURCE_DIR}
)