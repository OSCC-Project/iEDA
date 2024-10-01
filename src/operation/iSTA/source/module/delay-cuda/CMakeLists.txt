cmake_minimum_required(VERSION 3.15.0)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

if(NOT CUDAToolkit_FOUND)
    include(FindCUDAToolkit)
endif()

set(lib_name "delay-gpu")

add_compile_options("-fopenmp")

if(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    set(NVCC_FLAGS -O3)
    message(STATUS "${lib_name} compile cuda code in release mode")
else()
    set(NVCC_FLAGS -G)
    message(STATUS "${lib_name} compile cuda code in debug mode")
endif()

file(GLOB_RECURSE source_file LIST_DIRECTORIES false *.cpp *.c *.cuh *.h *.cu)

add_library(${lib_name}  STATIC ${source_file})

target_include_directories(${lib_name} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_property(TARGET ${lib_name} PROPERTY LANGUAGES CXX CUDA)
set_target_properties(${lib_name} 
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(${lib_name} PRIVATE CUDA::cudart)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)
endif()
