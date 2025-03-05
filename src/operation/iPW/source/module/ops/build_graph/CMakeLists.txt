cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
add_library(build_graph ${SRC})

target_link_libraries(build_graph netlist liberty core usage)