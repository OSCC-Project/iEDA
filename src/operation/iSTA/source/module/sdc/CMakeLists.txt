cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

# SET(CMAKE_BUILD_TYPE "Release")
aux_source_directory(./ SRC)
add_library(sdc ${SRC})

target_link_libraries(sdc str tcl netlist)

