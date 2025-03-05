cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

# SET(CMAKE_BUILD_TYPE "Debug")
aux_source_directory(./ SRC)
add_library(sdc-cmd ${SRC})

target_link_libraries(sdc-cmd str netlist tcl sdc)

target_include_directories(sdc-cmd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)