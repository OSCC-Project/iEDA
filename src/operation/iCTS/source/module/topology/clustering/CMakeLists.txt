add_library(
  icts_source_module_topology_clustering
  ${ICTS_MODULE_TOPOLOGY}/clustering/Clustering.cc
)

target_link_libraries(
  icts_source_module_topology_clustering
  PUBLIC
  icts_source_database_spatial
  PRIVATE
  icts_source_database_design
  icts_source_module_topology_fast_clustering
  icts_source_module_topology_config
  icts_source_module_topology_cluster_constraints
  icts_source_module_topology_kmeans
  icts_source_module_topology_mcf
  icts_source_utils_geometry
)

target_include_directories(
  icts_source_module_topology_clustering
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_MODULE_TOPOLOGY}/clustering
)

if(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY_CLUSTERING)
  if(MSVC)
    target_compile_options(icts_source_module_topology_clustering PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_topology_clustering PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_topology_clustering PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
