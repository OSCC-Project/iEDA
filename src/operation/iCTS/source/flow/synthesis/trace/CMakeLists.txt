set(ICTS_FLOW_SYNTHESIS_TRACE_DISTANCE ${ICTS_FLOW_SYNTHESIS_TRACE}/distance)
set(ICTS_FLOW_SYNTHESIS_TRACE_DOMAIN_STATUS ${ICTS_FLOW_SYNTHESIS_TRACE}/domain_status)
set(ICTS_FLOW_SYNTHESIS_TRACE_LAYOUT ${ICTS_FLOW_SYNTHESIS_TRACE}/layout)
set(ICTS_FLOW_SYNTHESIS_TRACE_TOPOLOGY_BUILD ${ICTS_FLOW_SYNTHESIS_TRACE}/topology_build)

add_subdirectory(${ICTS_FLOW_SYNTHESIS_TRACE_DISTANCE})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_TRACE_DOMAIN_STATUS})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_TRACE_LAYOUT})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_TRACE_TOPOLOGY_BUILD})

add_library(icts_source_flow_synthesis_trace ${ICTS_FLOW_SYNTHESIS_TRACE}/SynthesisTrace.cc)

target_link_libraries(
  icts_source_flow_synthesis_trace
  PRIVATE
  icts_source_flow_synthesis_trace_domain_status
)

target_include_directories(
  icts_source_flow_synthesis_trace
  PUBLIC
  ${ICTS_FLOW}
)
