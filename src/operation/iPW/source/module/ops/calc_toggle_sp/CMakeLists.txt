cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

aux_source_directory(./ SRC)
add_library(calc_toggle_sp ${SRC})

target_link_libraries(calc_toggle_sp core)