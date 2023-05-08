CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET (CMAKE_CXX_STANDARD 20)
SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(CMAKE_BUILD_TYPE "Debug")
# MESSAGE(STATUS "iSTA Test")

FIND_PACKAGE(GTest REQUIRED)

AUX_SOURCE_DIRECTORY(. SourceFiles)
ADD_EXECUTABLE(iSTATest ${SourceFiles})

TARGET_LINK_LIBRARIES(iSTATest ista-engine
sdc-cmd shell-cmd sta log str time netlist liberty
delay utility sta-solver verilog-parser graph idb
tcl usage pthread stdc++fs IdbBuilder def_service lef_service gtest gtest_main)
