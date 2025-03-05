cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

aux_source_directory(./ SRC)
add_library(dump ${SRC})

target_link_libraries(dump core yaml-cpp)