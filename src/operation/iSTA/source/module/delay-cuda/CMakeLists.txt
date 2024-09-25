cmake_minimum_required(VERSION 3.15.0)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 17)

set(lib_name "delay_gpu")

aux_source_directory(./ SRC)

add_library(${lib_name} ${SRC})

set_property(TARGET ${lib_name} PROPERTY LANGUAGES CXX CUDA)

target_link_libraries(${lib_name} PRIVATE CUDA::cudart)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)