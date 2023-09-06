cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

aux_source_directory(./ SRC)

add_library(app ${SRC})

target_link_libraries(app core vcd_wrapper build_graph propagate_toggle_sp levelize dump calc_power plot_power)