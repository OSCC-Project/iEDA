add_library(icts_source_flow_optimization_model INTERFACE)

target_link_libraries(
  icts_source_flow_optimization_model
  INTERFACE
  icts_source_database_adapter_fast_sta
  icts_source_module_routing
)

target_include_directories(
  icts_source_flow_optimization_model
  INTERFACE
  ${ICTS_FLOW}
)
