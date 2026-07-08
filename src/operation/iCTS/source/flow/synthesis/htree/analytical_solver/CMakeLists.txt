set(ICTS_FLOW_SYNTHESIS_HTREE_ANALYTICAL_SOLVER_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyticalSolver.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/candidate/AnalyticalCandidate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/model/AnalyticalSolveProblem.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/model/AnalyticalSolverModel.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/model/MathHtreeModel.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/model/MathHtreeProblemBuilder.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/selection/AnalyticalSelection.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/selection/AnalyticalValidation.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/solver/HighsMathHtreeSolver.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/solver/MathHtreeMilpSolver.cc
)

add_library(icts_source_flow_synthesis_htree_analytical_solver ${ICTS_FLOW_SYNTHESIS_HTREE_ANALYTICAL_SOLVER_SRC})

target_link_libraries(
  icts_source_flow_synthesis_htree_analytical_solver
  PUBLIC
  icts_source_flow_synthesis_htree_topology_pruning
  icts_source_database_spatial
  PRIVATE
  icts_source_flow_synthesis_htree_compensation
  icts_source_flow_synthesis_htree_plan
  icts_source_flow_synthesis_htree_region
  icts_source_module_analytical_characterization
  icts_source_utils_logger
  log
  highs
)

target_include_directories(
  icts_source_flow_synthesis_htree_analytical_solver
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)

if(DEBUG_ICTS_SOURCE_FLOW_SYNTHESIS_HTREE_ANALYTICAL_SOLVER)
  if(MSVC)
    target_compile_options(icts_source_flow_synthesis_htree_analytical_solver PRIVATE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_flow_synthesis_htree_analytical_solver PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
