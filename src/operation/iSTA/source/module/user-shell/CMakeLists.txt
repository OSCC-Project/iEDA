CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET (CMAKE_CXX_STANDARD 20)

AUX_SOURCE_DIRECTORY(./ SRC)
ADD_LIBRARY(user-shell STATIC ${SRC})

target_link_libraries(user-shell str tcl log readline shell-cmd)