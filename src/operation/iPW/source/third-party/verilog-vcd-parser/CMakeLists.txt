cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-w)

include_directories(src/)

add_subdirectory(src)

# add_executable(vcd-parser ./vcdtool/vcdtool.cpp)

# target_link_libraries(vcd-parser verilog-vcd-parser)