cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")

add_subdirectory(source/iir-rust)
add_subdirectory(source/ir-solver)

aux_source_directory(./ SRC)

add_executable(iIR ${SRC})


