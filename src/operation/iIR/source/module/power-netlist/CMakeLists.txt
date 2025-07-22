cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

include(${HOME_CMAKE}/operation/idb.cmake)

aux_source_directory(./ SRC)
add_library(pg-netlist ${SRC})

target_link_libraries(pg-netlist idb def_service lef_service iIR-Rust)

target_include_directories(
  pg-netlist
  PUBLIC ${HOME_UTILITY}/stdBase/include ${HOME_DATABASE}/data/design
         ${HOME_DATABASE}/manager/builder)
