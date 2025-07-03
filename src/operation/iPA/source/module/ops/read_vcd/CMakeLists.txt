cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

aux_source_directory(./ SRC)
add_library(vcd_wrapper ${SRC})

target_link_libraries(vcd_wrapper annotate vcd)
