cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)

# find_package(BISON 3.0.4 REQUIRED)
# find_package(FLEX 2.5.35 REQUIRED)

# set(FLEX_INPUT  VCDScanner.l)
# set(FLEX_OUTPUT VCDScanner.cpp)
# set(FLEX_HEADER VCDScanner.hpp)

# set(BISON_INPUT  VCDParser.ypp)
# set(BISON_OUTPUT VCDParser.cpp)
# set(BISON_HEADER VCDParser.hpp)

#add_custom_command(
#  COMMAND ${BISON_EXECUTABLE}
#  ARGS -v --defines=${BISON_HEADER} ${BISON_INPUT} -o ${BISON_OUTPUT}
#  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${BISON_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/${BISON_HEADER}
#  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#  DEPENDS ${BISON_INPUT}
#)
#
#add_custom_command(
#  COMMAND ${FLEX_EXECUTABLE}
#  ARGS -P VCDParser --never-interactive --header-file=${FLEX_HEADER} -o ${FLEX_OUTPUT} ${FLEX_INPUT}
#  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${FLEX_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/${FLEX_HEADER}
#  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#  DEPENDS ${BISON_OUTPUT} ${FLEX_INPUT}
#)

aux_source_directory(./ SRC)

find_package(ZLIB)

add_library(verilog-vcd-parser ${SRC})

target_link_libraries(verilog-vcd-parser PRIVATE ZLIB::ZLIB)