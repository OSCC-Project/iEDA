add_library(
  icts_source_module_topology_fast_clustering
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/FastClustering.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/boundary/FastClusteringBoundaryCandidates.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/boundary/FastClusteringBoundaryPolish.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/boundary/FastClusteringBoundarySearch.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/cluster_draft/FastClusteringDraft.hh
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/finalize/FastClusteringFinalize.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/geometry/FastClusteringGeometry.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/partition/FastClusteringPartition.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/polish/FastClusteringMergePolish.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/polish/FastClusteringPolish.cc
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/polish/FastClusteringPolishShared.cc
)

target_link_libraries(
  icts_source_module_topology_fast_clustering
  PUBLIC
  icts_source_database_spatial
  icts_source_module_topology_config
  PRIVATE
  icts_source_database_design
  icts_source_module_topology_cluster_constraints
  icts_source_utils_geometry
  log
)

target_include_directories(
  icts_source_module_topology_fast_clustering
  PUBLIC
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering
  PRIVATE
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/boundary
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/cluster_draft
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/finalize
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/geometry
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/partition
  ${ICTS_MODULE_TOPOLOGY}/fast_clustering/polish
  ${ICTS_MODULE_TOPOLOGY}/clustering
)

if(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY_FAST_CLUSTERING)
  if(MSVC)
    target_compile_options(icts_source_module_topology_fast_clustering PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_topology_fast_clustering PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_topology_fast_clustering PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
