add_subdirectory(${ICTS_SOLVER}/GOCA/tools/balance_clustering)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools/timing_propagator)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools/tree_builder)

add_library(icts_goca_tools INTERFACE)

target_link_libraries(
  icts_goca_tools INTERFACE icts_goca_balance_clustering
                            icts_goca_timing_propagator icts_goca_tree_builder)

target_include_directories(
  icts_goca_tools
  INTERFACE ${ICTS_SOLVER}/GOCA/tools/balance_clustering
            ${ICTS_SOLVER}/GOCA/tools/timing_propagator
            ${ICTS_SOLVER}/GOCA/tools/tree_builder)
