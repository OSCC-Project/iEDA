add_library(icts_source_flow_synthesis_trace_domain_status ${ICTS_FLOW_SYNTHESIS_TRACE_DOMAIN_STATUS}/DomainStatus.cc)

target_link_libraries(
  icts_source_flow_synthesis_trace_domain_status
  PUBLIC
  icts_source_utils_logger
  PRIVATE
  icts_source_database_design
)

target_include_directories(
  icts_source_flow_synthesis_trace_domain_status
  PUBLIC
  ${ICTS_FLOW}
)
