cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

aux_source_directory(./ SRC)

add_library(power ${SRC})

target_link_libraries(power core vcd_wrapper build_graph propagate_toggle_sp levelize dump calc_power plot_power)
target_include_directories(power PUBLIC ${PROJECT_SOURCE_DIR}/src/operation/iPW/source/module)