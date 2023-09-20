if(DEBUG_ICTS_ANNEAL_OPT)
  message(STATUS "CTS: DEBUG_ICTS_ANNEAL_OPT")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_ANNEAL_OPT")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(
  icts_anneal_opt
  ${ICTS_SOLVER}/GOCA/tools/balance_clustering/anneal_opt/AnnealOpt.cc)

target_link_libraries(icts_anneal_opt PUBLIC icts_goca_database
                                             icts_timing_propagator)

target_include_directories(
  icts_anneal_opt PUBLIC ${ICTS_SOLVER}/GOCA/database
                         ${ICTS_SOLVER}/GOCA/tools/timing_propagator)
