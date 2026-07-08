add_library(
  icts_test_common_topology_analysis
  STATIC
  ${ICTS_TEST}/common/topology/TopologyAnalysis.cc
)

target_link_libraries(
  icts_test_common_topology_analysis
  PRIVATE
  icts_test_common_dataset
)
