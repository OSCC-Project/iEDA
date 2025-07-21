cmake_minimum_required(VERSION 3.11)
set (CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
add_library(ista_utility ${SRC})

target_link_libraries(ista_utility log)
