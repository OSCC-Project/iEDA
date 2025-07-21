cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

if(NOT DEFINED CMD_BUILD)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

aux_source_directory(./ SRC)

add_library(delay ${SRC})

target_link_libraries(delay spef str ista_utility liberty netlist Eigen3::Eigen log) 
