cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)

# set(CMAKE_BUILD_TYPE "Release")

aux_source_directory(./ SRC)
add_library(sta ${SRC})

find_package(yaml-cpp REQUIRED)

target_link_libraries(sta liberty delay sdc sdc-cmd verilog-parser aocv-parser graph sdc absl::btree tcl time report_table stdc++fs log yaml-cpp)

