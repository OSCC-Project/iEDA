CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET (CMAKE_CXX_STANDARD 20)

# SET(CMAKE_BUILD_TYPE "Release")
AUX_SOURCE_DIRECTORY(./ SRC)
ADD_LIBRARY(pwr-cmd ${SRC})

target_link_libraries(pwr-cmd tcl power usage)

target_include_directories(pwr-cmd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)