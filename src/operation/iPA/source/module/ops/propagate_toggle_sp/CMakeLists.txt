cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

if(NOT DEFINED CMD_BUILD)
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
endif()

aux_source_directory(./ SRC)
add_library(propagate_toggle_sp ${SRC})

target_link_libraries(propagate_toggle_sp core calc_toggle_sp dump)