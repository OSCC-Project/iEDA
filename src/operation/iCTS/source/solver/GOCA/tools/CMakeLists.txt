add_subdirectory(${ICTS_SOLVER}/GOCA/tools/balance_clustering)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools/timing_propagator)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools/tree_builder)

add_library(icts_tools INTERFACE)

target_link_libraries(
  icts_tools INTERFACE icts_balance_clustering icts_timing_propagator
                       icts_tree_builder)

target_include_directories(
  icts_tools
  INTERFACE ${ICTS_SOLVER}/GOCA/tools/balance_clustering
            ${ICTS_SOLVER}/GOCA/tools/timing_propagator
            ${ICTS_SOLVER}/GOCA/tools/tree_builder)
