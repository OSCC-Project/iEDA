add_library(
  icts_test_module_topology_gen_artifact_writer
  STATIC
  ${ICTS_TEST}/module/topology/topology_gen/fixture/artifact/TopologyGenArtifactWriter.cc
)

target_link_libraries(
  icts_test_module_topology_gen_artifact_writer
  PUBLIC
  icts_test_common_topology_analysis
  icts_test_common_visualization
  PRIVATE
  icts_test_base
)
