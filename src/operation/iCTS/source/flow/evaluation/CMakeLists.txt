set(ICTS_FLOW_EVALUATION_QOR ${ICTS_FLOW_EVALUATION}/qor)

add_subdirectory(${ICTS_FLOW_EVALUATION_QOR})

add_library(icts_source_flow_evaluation ${ICTS_FLOW_EVALUATION}/Evaluation.cc)

target_link_libraries(
  icts_source_flow_evaluation
  PUBLIC
  icts_source_flow_evaluation_qor
  PRIVATE
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_evaluation
  PUBLIC
  ${ICTS_FLOW}
)
