set(ICTS_FLOW_EVALUATION ${ICTS_FLOW}/evaluation)
set(ICTS_FLOW_INSTANTIATION ${ICTS_FLOW}/instantiation)
set(ICTS_FLOW_OPTIMIZATION ${ICTS_FLOW}/optimization)
set(ICTS_FLOW_REPORT ${ICTS_FLOW}/report)
set(ICTS_FLOW_SETUP ${ICTS_FLOW}/setup)
set(ICTS_FLOW_SYNTHESIS ${ICTS_FLOW}/synthesis)

add_subdirectory(${ICTS_FLOW_SETUP})
add_subdirectory(${ICTS_FLOW_SYNTHESIS})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION})
add_subdirectory(${ICTS_FLOW_INSTANTIATION})
add_subdirectory(${ICTS_FLOW_EVALUATION})
add_subdirectory(${ICTS_FLOW_REPORT})

add_library(icts_source_flow ${ICTS_FLOW}/Flow.cc)

target_link_libraries(
  icts_source_flow
  PUBLIC
  icts_source_flow_instantiation
  icts_source_flow_synthesis
  icts_source_flow_optimization
  icts_source_flow_evaluation
  icts_source_module_characterization_headers
  icts_source_database_adapter_fast_sta
  icts_source_database_config
  icts_source_database_qor
  PRIVATE
  icts_source_flow_report
  icts_source_flow_setup
  icts_source_flow_setup_clock_data
  icts_source_module_routing
  icts_source_module_topology_clustering
  log
)

target_include_directories(
  icts_source_flow
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)

if(DEBUG_ICTS_SOURCE_FLOW)
  if(MSVC)
    target_compile_options(icts_source_flow PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_flow PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_flow PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
