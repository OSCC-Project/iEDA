CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET (CMAKE_CXX_STANDARD 20)

# SET(CMAKE_BUILD_TYPE "Release")

AUX_SOURCE_DIRECTORY(./ SRC)
ADD_LIBRARY(sta ${SRC})

FIND_PACKAGE(yaml-cpp REQUIRED)

TARGET_LINK_LIBRARIES(sta liberty delay sdc sdc-cmd verilog-parser aocv-parser graph sdc absl::btree tcl time report_table stdc++fs log yaml-cpp)

