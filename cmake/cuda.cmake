macro(ADD_CUDA_PROJ proj_name)

cmake_minimum_required(VERSION 3.15.0)

cmake_policy(SET CMP0128 NEW)

set(CMAKE_CUDA_ARCHITECTURES native)
# set(CMAKE_CUDA_ARCHITECTURES "70") # set architecture according your platform

# set(CMAKE_BUILD_TYPE "Debug")
find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 17)
add_compile_options(-std=c++17)

set(CMAKE_CUDA_FLAGS_INIT "-std=c++17")

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

if(NOT CUDAToolkit_FOUND)
    include(FindCUDAToolkit)
endif()

set(lib_name ${proj_name})

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(NVCC_FLAGS -O3)
    message(STATUS "${lib_name} compile cuda code in release mode")
else()
    set(NVCC_FLAGS -G -g)
    message(STATUS "${lib_name} compile cuda code in debug mode")
endif()

file(GLOB_RECURSE source_file LIST_DIRECTORIES false *.cpp *.c *.cuh *.h *.cu)

if(BUILD_STATIC_LIB)
add_library(${lib_name} ${source_file})
else()
add_library(${lib_name} SHARED ${source_file})
endif()

set_property(TARGET ${lib_name} PROPERTY LANGUAGES CXX CUDA)
set_target_properties(${lib_name} 
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${lib_name} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)
target_link_libraries(${lib_name} PRIVATE CUDA::cudart)

endmacro()