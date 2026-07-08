add_subdirectory(analytical_solver)

icts_add_test_executable(
  icts_test_flow_synthesis_htree
  PUBLIC_RUNTIME
  SOURCES
    ${ICTS_TEST}/flow/synthesis/htree/HTreeTest.cc
  LIBS
    icts_test_common_data_pin_factory
)

if(ICTS_BUILD_REALTECH_TESTS)
  icts_add_test_executable(
    icts_test_flow_synthesis_htree_realtech
    REALTECH
    SOURCES
      ${ICTS_TEST}/flow/synthesis/htree/HTreeRealTechScenario.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeVisualizationDelayPower.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeVisualizationRendering.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeArtifactWriter.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeRealTechMatrixRunner.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeRealTechMatrixTest.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeRealTechSmokeTest.cc
    LIBS
      icts_test_common_data_pin_factory
      icts_test_common_scoped_log_file
      icts_test_common_io
      icts_test_common_realtech_setup
      icts_test_common_visualization
      icts_test_module_characterization_fixture
  )

  target_compile_definitions(
    icts_test_flow_synthesis_htree_realtech
    PRIVATE
      ICTS_ENABLE_SLOW_REALTECH_REGRESSION=0
  )
endif()

if(ICTS_BUILD_REALTECH_TESTS AND ICTS_BUILD_SLOW_REALTECH_TESTS)
  icts_add_test_executable(
    icts_test_flow_synthesis_htree_realtech_regression
    REALTECH
    SOURCES
      ${ICTS_TEST}/flow/synthesis/htree/HTreeRealTechScenario.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeVisualizationDelayPower.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeVisualizationRendering.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeArtifactWriter.cc
      ${ICTS_TEST}/flow/synthesis/htree/HTreeRealTechBranchBufferRegressionTest.cc
    LIBS
      icts_test_common_data_pin_factory
      icts_test_common_scoped_log_file
      icts_test_common_io
      icts_test_common_realtech_setup
      icts_test_common_visualization
      icts_test_module_characterization_fixture
  )

  target_compile_definitions(
    icts_test_flow_synthesis_htree_realtech_regression
    PRIVATE
      ICTS_ENABLE_SLOW_REALTECH_REGRESSION=1
  )
endif()
