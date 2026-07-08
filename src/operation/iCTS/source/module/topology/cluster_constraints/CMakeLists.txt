add_library(
  icts_source_module_topology_cluster_constraints
  ${ICTS_MODULE_TOPOLOGY}/cluster_constraints/ClusterConstraintEvaluator.cc
)

target_link_libraries(
  icts_source_module_topology_cluster_constraints
  PUBLIC
  icts_source_database_spatial
  PRIVATE
  icts_source_database_design
  icts_source_module_routing
  icts_source_module_routing_helper
  icts_source_module_timing
  icts_source_module_topology_config
  log
)

target_include_directories(
  icts_source_module_topology_cluster_constraints
  PUBLIC
  ${ICTS_MODULE_TOPOLOGY}/cluster_constraints
)
