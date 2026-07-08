add_library(icts_source_module_topology_kmeans INTERFACE)

target_link_libraries(
  icts_source_module_topology_kmeans
  INTERFACE
  icts_source_database_spatial
  icts_source_utils_geometry
)

target_include_directories(
  icts_source_module_topology_kmeans
  INTERFACE
  ${ICTS_MODULE_TOPOLOGY}/kmeans
)

if(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY_KMEANS)
  if(MSVC)
    target_compile_options(icts_source_module_topology_kmeans INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_module_topology_kmeans INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
