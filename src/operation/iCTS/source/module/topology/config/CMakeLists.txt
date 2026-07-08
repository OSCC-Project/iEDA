add_library(icts_source_module_topology_config INTERFACE)

target_link_libraries(
  icts_source_module_topology_config
  INTERFACE
  icts_source_database_routing
)

target_include_directories(
  icts_source_module_topology_config
  INTERFACE
  ${ICTS_MODULE_TOPOLOGY}/config
)
