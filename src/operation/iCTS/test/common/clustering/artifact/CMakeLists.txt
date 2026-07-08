add_library(
  icts_test_common_clustering_artifact
  STATIC
  ${ICTS_TEST}/common/clustering/artifact/ClusterArtifactWriter.cc
)

target_link_libraries(
  icts_test_common_clustering_artifact
  PRIVATE
  icts_test_common_clustering_metrics
  icts_source_module_topology_clustering
)
