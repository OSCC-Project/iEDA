CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET (CMAKE_CXX_STANDARD 20)

# SET(CMAKE_BUILD_TYPE "Release")
AUX_SOURCE_DIRECTORY(./ SRC)
ADD_LIBRARY(sdc-cmd ${SRC})

target_link_libraries(sdc-cmd str netlist tcl sdc)

target_include_directories(sdc-cmd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)