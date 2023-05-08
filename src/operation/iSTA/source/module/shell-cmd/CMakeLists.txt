CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET (CMAKE_CXX_STANDARD 20)

AUX_SOURCE_DIRECTORY(./ SRC)
ADD_LIBRARY(shell-cmd ${SRC})

target_link_libraries(shell-cmd str tcl netlist sta)

target_include_directories(shell-cmd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)