cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Debug")

aux_source_directory(./ SRC)
add_library(sta ${SRC})

target_link_libraries(sta 
    liberty 
    delay 
    spef 
    sdc 
    sdc-cmd 
    verilog 
    aocv-parser 
    graph sdc 
    absl::btree 
    tcl 
    time 
    report_table 
    stdc++fs 
    log
    yaml-cpp)

if (USE_CUDA_MODE)
target_link_libraries(sta
    delay-gpu 
    propagation-gpu
)
endif()