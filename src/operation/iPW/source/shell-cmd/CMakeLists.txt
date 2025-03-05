cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")
aux_source_directory(./ SRC)
add_library(pwr-cmd ${SRC})

target_link_libraries(pwr-cmd tcl power sta usage)

target_include_directories(pwr-cmd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)