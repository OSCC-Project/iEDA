add_library(
  icts_test_common_clustering_metrics
  STATIC
  ${ICTS_TEST}/common/clustering/metrics/ClusterGeometryMetrics.cc
)

target_link_libraries(
  icts_test_common_clustering_metrics
  PUBLIC
  icts_test_base
  PRIVATE
  icts_source_module_topology_clustering
)
