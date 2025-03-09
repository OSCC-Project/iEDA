cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

if(NOT DEFINED CMD_BUILD)
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

aux_source_directory(./ SRC)
add_library(iir-api ${SRC} )

target_link_libraries(iir-api ir_solver ir-matrix Eigen3::Eigen pg-netlist)
