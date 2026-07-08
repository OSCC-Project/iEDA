set(ICTS_FLOW_OPTIMIZATION_CANDIDATE ${ICTS_FLOW_OPTIMIZATION}/candidate)
set(ICTS_FLOW_OPTIMIZATION_MODEL ${ICTS_FLOW_OPTIMIZATION}/model)
set(ICTS_FLOW_OPTIMIZATION_CLOCK_SIZING_EDIT ${ICTS_FLOW_OPTIMIZATION}/clock_sizing_edit)
set(ICTS_FLOW_OPTIMIZATION_POLICY ${ICTS_FLOW_OPTIMIZATION}/policy)
set(ICTS_FLOW_OPTIMIZATION_PREPARATION ${ICTS_FLOW_OPTIMIZATION}/preparation)
set(ICTS_FLOW_OPTIMIZATION_REPORT ${ICTS_FLOW_OPTIMIZATION}/report)
set(ICTS_FLOW_OPTIMIZATION_SOLVER ${ICTS_FLOW_OPTIMIZATION}/solver)
set(ICTS_FLOW_OPTIMIZATION_STATE ${ICTS_FLOW_OPTIMIZATION}/state)

add_subdirectory(${ICTS_FLOW_OPTIMIZATION_MODEL})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION_POLICY})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION_REPORT})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION_PREPARATION})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION_CANDIDATE})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION_STATE})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION_SOLVER})
add_subdirectory(${ICTS_FLOW_OPTIMIZATION_CLOCK_SIZING_EDIT})

add_library(
  icts_source_flow_optimization
  ${ICTS_FLOW_OPTIMIZATION}/Optimization.cc
)

target_link_libraries(
  icts_source_flow_optimization
  PRIVATE
  icts_source_database_adapter_fast_sta
  icts_source_database_config
  icts_source_database_io
  icts_source_module_routing
  icts_source_database_design
  icts_source_flow_optimization_model
  icts_source_flow_optimization_clock_sizing_edit
  icts_source_flow_optimization_policy
  icts_source_flow_optimization_preparation
  icts_source_flow_optimization_report
  icts_source_flow_optimization_solver
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_optimization
  PUBLIC
  ${ICTS_FLOW}
)

if(DEBUG_ICTS_SOURCE_FLOW_OPTIMIZATION)
  if(MSVC)
    target_compile_options(icts_source_flow_optimization PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_flow_optimization PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_flow_optimization PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
