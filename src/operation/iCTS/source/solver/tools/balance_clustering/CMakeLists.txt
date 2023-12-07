add_subdirectory(${ICTS_SOLVER}/tools/balance_clustering/min_cost_flow)
add_subdirectory(${ICTS_SOLVER}/tools/balance_clustering/anneal_opt)

if(DEBUG_ICTS_BALANCE_CLUSTERING)
  message(STATUS "CTS: DEBUG_ICTS_BALANCE_CLUSTERING")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_BALANCE_CLUSTERING")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_balance_clustering
            ${ICTS_SOLVER}/tools/balance_clustering/BalanceClustering.cc)

target_link_libraries(
  icts_balance_clustering
  PUBLIC icts_data_manager
         icts_api
         icts_solver_database
         icts_min_cost_flow
         icts_anneal_opt
         icts_timing_propagator
         icts_tree_builder)

target_include_directories(
  icts_balance_clustering
  PUBLIC ${ICTS_SOLVER}/database
         ${ICTS_SOLVER}/tools/balance_clustering/min_cost_flow
         ${ICTS_SOLVER}/tools/balance_clustering/simulated_annealing
         ${ICTS_SOLVER}/tools/timing_propagator
         ${ICTS_SOLVER}/tools/tree_builder
         ${ICTS_DATABASE})
