cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)

SET(CMAKE_BUILD_TYPE "Debug")

find_package(GTest REQUIRED)

aux_source_directory(./ SRC)

add_executable(iPowerTest ${SRC})

target_link_libraries(
  iPowerTest
  PUBLIC
  gtest
  gtest_main
  boost_system
  PRIVATE
  vcd_wrapper
  log
  power
  usage ista-engine core calc_power plot_power

)
