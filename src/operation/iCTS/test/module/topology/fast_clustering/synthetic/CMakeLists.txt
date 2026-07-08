icts_add_test_executable(
  icts_test_module_topology_fast_clustering
  SOURCES
    ${ICTS_TEST}/module/topology/fast_clustering/synthetic/FastClusteringSyntheticTest.cc
  LIBS
    icts_test_common_data_pin_factory
    icts_source_module_topology
)
