add_library(
  icts_test_module_topology_gen_scenario_runner
  STATIC
  ${ICTS_TEST}/module/topology/topology_gen/fixture/core/TopologyGenScenarioRunner.cc
)

target_link_libraries(
  icts_test_module_topology_gen_scenario_runner
  PUBLIC
  icts_test_module_topology_gen_case_factory
  icts_test_module_topology_gen_tree_assertions
  icts_test_module_topology_gen_artifact_writer
  PRIVATE
  log
  icts_test_common_io
  icts_test_common_scoped_log_file
)
