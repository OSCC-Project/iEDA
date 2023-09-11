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

add_library(verilog-parser ${SRC} 
${HOME_THIRDPARTY}/parser/verilog/mVerilogParse.cc 
${HOME_THIRDPARTY}/parser/verilog/mVerilogLex.cc)

target_link_libraries(verilog-parser str log ${ZLIB_LIBRARIES}) 

target_include_directories(verilog-parser 
    PUBLIC
        ${HOME_OPERATION}/iSTA/include  
        ${HOME_THIRDPARTY}/parser/verilog
        ${CMAKE_CURRENT_SOURCE_DIR}
)