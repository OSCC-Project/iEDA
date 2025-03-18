cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_TYPE "Debug")

find_package(GTest REQUIRED)

aux_source_directory(. SourceFiles)
add_executable(iIRTest ${SourceFiles})

target_link_libraries(iIRTest iir-api iIR-Rust gtest log)
