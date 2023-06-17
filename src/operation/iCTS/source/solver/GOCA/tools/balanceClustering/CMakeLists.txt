add_subdirectory(${ICTS_SOLVER}/GOCA/tools/balanceClustering/minCostFlow)

if(DEBUG_ICTS_GOCA_BALANCE_CLUSTERING)
  message(STATUS "CTS: DEBUG_ICTS_GOCA_BALANCE_CLUSTERING")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_GOCA_BALANCE_CLUSTERING")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_goca_balance_clustering
            ${ICTS_SOLVER}/GOCA/tools/balanceClustering/BalanceClustering.cc)

target_link_libraries(icts_goca_balance_clustering
                      PUBLIC icts_data_manager icts_api icts_min_cost_flow)

target_include_directories(
  icts_goca_balance_clustering
  PUBLIC ${ICTS_SOLVER}/GOCA/database ${ICTS_SOLVER}/GOCA/tools/minCostFlow
         ${ICTS_DATABASE})
