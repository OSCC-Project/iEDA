cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

aux_source_directory(./ SRC)
add_library(ir-matrix ${SRC} )

target_link_libraries(ir-matrix iIR-Rust Eigen3::Eigen)