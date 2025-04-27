cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

# set(CMAKE_BUILD_TYPE "Debug")
aux_source_directory(./ SRC)

add_library(netlist ${SRC})

target_link_libraries(netlist str liberty log absl::inlined_vector)
