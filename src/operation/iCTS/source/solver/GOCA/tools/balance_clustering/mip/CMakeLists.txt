if(DEBUG_ICTS_MIP)
  message(STATUS "CTS: DEBUG_ICTS_MIP")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_MIP")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(TARGET SCIP::SCIP)
  # find package by SCIP PATH
  find_package(SCIP CONFIG PATHS ${SCIP_BINARY_DIR} REQUIRED)
else()
  find_package(SCIP REQUIRED)
endif()

add_library(icts_mip ${ICTS_SOLVER}/GOCA/tools/balance_clustering/mip/MIP.cc)

target_link_libraries(icts_mip PUBLIC icts_goca_database icts_timing_propagator
                                      ${SCIP_LIBRARIES})

target_include_directories(
  icts_mip
  PUBLIC ${ICTS_SOLVER}/GOCA/database
         ${ICTS_SOLVER}/GOCA/tools/timing_propagator ${SCIP_INCLUDE_DIRS})
