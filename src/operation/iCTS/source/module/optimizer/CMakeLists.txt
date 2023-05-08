if(DEBUG_ICTS_OPTIMIZER)
  message(STATUS "CTS: DEBUG_ICTS_OPTIMIZER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_OPTIMIZER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_optimizer ${ICTS_MODULE}/optimizer/Optimizer.cc
                           ${ICTS_MODULE}/optimizer/OptiNet.cc)

target_link_libraries(icts_optimizer PUBLIC icts_data_manager icts_module)

target_include_directories(icts_optimizer PUBLIC ${ICTS_MODULE}/optimizer)
