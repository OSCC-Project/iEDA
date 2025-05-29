cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

if(NOT DEFINED CMD_BUILD)
    set (CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")
endif()

include_directories(${HOME_OPERATION}/iIR)
include_directories(${HOME_OPERATION}/iIR/source)
include_directories(${HOME_OPERATION}/iIR/source/module)
include_directories(${HOME_THIRDPARTY}/spectra/include)

if (USE_GPU)
  find_package(CUDAToolkit)

  if(CUDAToolkit_FOUND)
      set(CUDA_IR_SOLVER ON)
      if (CUDA_IR_SOLVER)
          message(STATUS "CUDAToolkit is available, use cuda ir solver.")
          add_definitions(-DCUDA_IR_SOLVER=1)
          add_subdirectory(source/ir-solver-cuda)
      endif()
  else()
      message("CUDAToolkit is not available.")
  endif()
endif()

add_subdirectory(api)
add_subdirectory(source/iir-rust)
add_subdirectory(source/ir-solver)
add_subdirectory(source/module/matrix)
add_subdirectory(source/module/power-netlist)
add_subdirectory(test)

aux_source_directory(./ SRC)

add_executable(iIR ${SRC})


