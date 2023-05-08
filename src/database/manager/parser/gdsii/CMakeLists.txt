##-------------------------- CMake Information ----------------------##
cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##-------------------------- Compile Options ------------------------##
# Configure compile options
add_compile_options(-Wall -Wextra -pedantic)

##-------------------------- Library --------------------------------##
aux_source_directory(. SRC)
add_library(gdsii-parser ${SRC})

target_include_directories(gdsii-parser 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
# SET(CMAKE_BUILD_TYPE "Debug")