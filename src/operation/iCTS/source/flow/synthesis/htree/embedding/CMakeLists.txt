add_library(icts_source_flow_synthesis_htree_embedding ${ICTS_FLOW_SYNTHESIS_HTREE_EMBEDDING}/Embedding.cc)

target_link_libraries(
  icts_source_flow_synthesis_htree_embedding
  PUBLIC
  icts_source_database_io
  PRIVATE
  icts_source_flow_synthesis_htree_segment_pruning
  icts_source_database_design
  icts_source_database_spatial
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_htree_embedding
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
