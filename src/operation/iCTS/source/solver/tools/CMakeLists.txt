add_subdirectory(${ICTS_SOLVER}/tools/balance_clustering)
add_subdirectory(${ICTS_SOLVER}/tools/timing_propagator)
add_subdirectory(${ICTS_SOLVER}/tools/tree_builder)

add_library(icts_tools INTERFACE)

target_link_libraries(
  icts_tools INTERFACE icts_balance_clustering icts_timing_propagator
                       icts_tree_builder)

target_include_directories(
  icts_tools
  INTERFACE ${ICTS_SOLVER}/tools/balance_clustering
            ${ICTS_SOLVER}/tools/timing_propagator
            ${ICTS_SOLVER}/tools/tree_builder)
