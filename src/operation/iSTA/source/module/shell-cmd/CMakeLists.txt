cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
add_library(shell-cmd ${SRC})

target_link_libraries(shell-cmd str tcl sta ista-engine)

target_include_directories(shell-cmd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)