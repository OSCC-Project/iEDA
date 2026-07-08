set(ICTS_FLOW_SYNTHESIS_HTREE_CHARACTERIZATION ${ICTS_FLOW_SYNTHESIS_HTREE}/characterization)
set(ICTS_FLOW_SYNTHESIS_HTREE_ANALYTICAL_SOLVER ${ICTS_FLOW_SYNTHESIS_HTREE}/analytical_solver)
set(ICTS_FLOW_SYNTHESIS_HTREE_COMPENSATION ${ICTS_FLOW_SYNTHESIS_HTREE}/compensation)
set(ICTS_FLOW_SYNTHESIS_HTREE_CONSTRAINT ${ICTS_FLOW_SYNTHESIS_HTREE}/constraint)
set(ICTS_FLOW_SYNTHESIS_HTREE_EMBEDDING ${ICTS_FLOW_SYNTHESIS_HTREE}/embedding)
set(ICTS_FLOW_SYNTHESIS_HTREE_PLAN ${ICTS_FLOW_SYNTHESIS_HTREE}/plan)
set(ICTS_FLOW_SYNTHESIS_HTREE_REGION ${ICTS_FLOW_SYNTHESIS_HTREE}/region)
set(ICTS_FLOW_SYNTHESIS_HTREE_SEGMENT_PRUNING ${ICTS_FLOW_SYNTHESIS_HTREE}/segment_pruning)
set(ICTS_FLOW_SYNTHESIS_HTREE_SOLUTION ${ICTS_FLOW_SYNTHESIS_HTREE}/solution)
set(ICTS_FLOW_SYNTHESIS_HTREE_SYNTHESIS_STATE ${ICTS_FLOW_SYNTHESIS_HTREE}/synthesis_state)
set(ICTS_FLOW_SYNTHESIS_HTREE_TOPOLOGY_PRUNING ${ICTS_FLOW_SYNTHESIS_HTREE}/topology_pruning)

add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_CHARACTERIZATION})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_ANALYTICAL_SOLVER})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_COMPENSATION})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_CONSTRAINT})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_EMBEDDING})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_PLAN})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_REGION})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_SEGMENT_PRUNING})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_TOPOLOGY_PRUNING})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_SYNTHESIS_STATE})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_HTREE_SOLUTION})

add_library(
  icts_source_flow_synthesis_htree
  ${ICTS_FLOW_SYNTHESIS_HTREE}/HTree.cc
)

target_link_libraries(
  icts_source_flow_synthesis_htree
  PUBLIC
  icts_source_database_design
  icts_source_database_spatial
  icts_source_database_characterization
  icts_source_module_characterization_headers
  PRIVATE
  icts_source_flow_synthesis_htree_solution
  icts_source_flow_synthesis_htree_synthesis_state
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_htree
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
