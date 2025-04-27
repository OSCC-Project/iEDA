cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_TYPE "Debug")
message(STATUS "iSTA Test")

find_package(GTest REQUIRED)

aux_source_directory(. SourceFiles)
add_executable(iSTATest ${SourceFiles})

target_link_libraries(
  iSTATest
  ista-engine
  inference
  sdc-cmd
  shell-cmd
  sta
  log
  str
  time
  netlist
  liberty
  delay
  ista_utility
  sta-solver
  verilog
  graph
  idb
  tcl
  usage
  pthread
  stdc++fs
  IdbBuilder
  def_service
  lef_service
  gtest)

if(USE_CUDA_MODE)
  target_link_libraries(iSTATest delay-gpu propagation-gpu)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(iSTATest
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)
endif()
