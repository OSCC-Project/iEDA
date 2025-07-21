cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

# set(CMAKE_BUILD_TYPE "Release")

aux_source_directory(./ SRC)
add_library(ista-engine ${SRC})

target_link_libraries(
  ista-engine
  sta
  str
  sdc
  sdc-cmd
  tcl
  idb
  def_service
  lef_service
  log
  absl::flat_hash_map)

target_include_directories(
  ista-engine
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${HOME_UTILITY}/stdBase/include
         ${HOME_OPERATION}/iSTA/source/module
         ${HOME_OPERATION}/iSTA/source/module/include)
