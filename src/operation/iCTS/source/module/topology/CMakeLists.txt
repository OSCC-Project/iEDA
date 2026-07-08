add_subdirectory(${ICTS_MODULE_TOPOLOGY}/config)
add_subdirectory(${ICTS_MODULE_TOPOLOGY}/kmeans)
add_subdirectory(${ICTS_MODULE_TOPOLOGY}/mcf)
add_subdirectory(${ICTS_MODULE_TOPOLOGY}/cluster_constraints)
add_subdirectory(${ICTS_MODULE_TOPOLOGY}/fast_clustering)
add_subdirectory(${ICTS_MODULE_TOPOLOGY}/clustering)

add_library(
  icts_source_module_topology
  ${ICTS_MODULE_TOPOLOGY}/TopologyGen.cc
)

target_link_libraries(
  icts_source_module_topology
  PUBLIC
  icts_source_database_spatial
  icts_source_module_topology_config
  PRIVATE
  icts_source_database_design
  icts_source_module_topology_clustering
  icts_source_module_topology_fast_clustering
  icts_source_utils_geometry
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_module_topology
  PUBLIC
  ${ICTS_MODULE_TOPOLOGY}
)

if(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY)
  if(MSVC)
    target_compile_options(icts_source_module_topology PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_topology PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_topology PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
