cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)
set(SPEF_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${SPEF_GENERATED_DIR})
find_package(FLEX REQUIRED)
flex_target(ircxSpefScanner ${CMAKE_CURRENT_SOURCE_DIR}/spef_lex.l
  ${SPEF_GENERATED_DIR}/spef_lex.cpp DEFINES_FILE ${SPEF_GENERATED_DIR}/spef_lex.hpp)
find_package(BISON REQUIRED)
bison_target(ircxSpefParser ${CMAKE_CURRENT_SOURCE_DIR}/spef_parser.yy
  ${SPEF_GENERATED_DIR}/spef_parser.cpp DEFINES_FILE ${SPEF_GENERATED_DIR}/spef_parser.hpp)
add_library(ircx_spef SpefParser.cc ${FLEX_ircxSpefScanner_OUTPUTS} ${BISON_ircxSpefParser_OUTPUTS})
target_link_libraries(ircx_spef log)
target_include_directories(ircx_spef PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${SPEF_GENERATED_DIR})
