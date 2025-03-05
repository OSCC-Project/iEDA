cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")

include_directories(${HOME_OPERATION}/iIR)
include_directories(${HOME_OPERATION}/iIR/source)
include_directories(${HOME_OPERATION}/iIR/source/module)

add_subdirectory(api)
add_subdirectory(source/iir-rust)
add_subdirectory(source/ir-solver)
add_subdirectory(source/module/matrix)
add_subdirectory(source/module/power-netlist)
add_subdirectory(test)

aux_source_directory(./ SRC)

add_executable(iIR ${SRC})


