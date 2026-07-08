add_library(
  icts_test_module_topology_gen_tree_assertions
  STATIC
  ${ICTS_TEST}/module/topology/topology_gen/fixture/analysis/TopologyGenTreeAssertions.cc
)

target_link_libraries(
  icts_test_module_topology_gen_tree_assertions
  PUBLIC
  icts_test_common_topology_analysis
  PRIVATE
  icts_test_base
)
