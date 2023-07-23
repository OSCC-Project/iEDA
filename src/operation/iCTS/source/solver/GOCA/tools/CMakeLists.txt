add_subdirectory(${ICTS_SOLVER}/GOCA/tools/balanceClustering)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools/timingPropagator)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools/treeBuilder)

add_library(icts_goca_tools INTERFACE)

target_link_libraries(
  icts_goca_tools INTERFACE icts_goca_balance_clustering
                            icts_goca_timing_propagator icts_goca_tree_builder)

target_include_directories(
  icts_goca_tools
  INTERFACE ${ICTS_SOLVER}/GOCA/tools/balanceClustering
            ${ICTS_SOLVER}/GOCA/tools/timingPropagator
            ${ICTS_SOLVER}/GOCA/tools/treeBuilder)
