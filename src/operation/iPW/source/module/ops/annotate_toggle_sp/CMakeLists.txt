cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
add_library(annotate ${SRC})

target_link_libraries(annotate gmp core)

