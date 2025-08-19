cmake_minimum_required(VERSION 3.11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD 20)

set(SANITIZER OFF)
if(SANITIZER)
  add_compile_options("-fsanitize=thread")
  link_libraries("-fsanitize=thread")
endif()

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# for sanitize SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb
# -fsanitize=thread -std=c++20")

add_compile_options("-march=native")

include_directories(${HOME_UTILITY}/stdBase)
include_directories(${HOME_UTILITY}/stdBase/include)
include_directories(${HOME_UTILITY})
include_directories(${HOME_DATABASE}/manager/parser)
include_directories(SYSTEM ${HOME_THIRDPARTY})
include(${HOME_CMAKE}/operation/ista.cmake)

include_directories(${PROJECT_SOURCE_DIR}/src/operation)
include_directories(${PROJECT_SOURCE_DIR}/src/operation/iPA)
include_directories(${PROJECT_SOURCE_DIR}/src/operation/iPA/source)
include_directories(${PROJECT_SOURCE_DIR}/src/operation/iPA/source/module)

include_directories(${HOME_THIRDPARTY}/pybind11)

if(USE_GPU)
  message(STATUS "power use gpu mode.")
  add_subdirectory(source/gpu-kernel)
endif()

add_subdirectory(api)
add_subdirectory(source/module/core)
add_subdirectory(source/module/ops/annotate_toggle_sp)
add_subdirectory(source/module/ops/build_graph)
add_subdirectory(source/module/ops/calc_power)
add_subdirectory(source/module/ops/calc_toggle_sp)
add_subdirectory(source/module/ops/dump)
add_subdirectory(source/module/ops/levelize_seq_graph)
add_subdirectory(source/module/ops/propagate_toggle_sp)
add_subdirectory(source/module/ops/read_vcd)
add_subdirectory(source/module/ops/plot_power)
add_subdirectory(source/shell-cmd)
add_subdirectory(source/python-api)

add_subdirectory(test)

add_executable(iPower main.cc)
target_link_libraries(
  iPower
  vcd_wrapper
  log
  power
  usage
  ista-engine
  core
  calc_power
  plot_power
  shell-cmd
  pwr-cmd)
