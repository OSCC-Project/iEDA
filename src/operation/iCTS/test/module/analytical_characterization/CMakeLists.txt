icts_add_test_executable(
  icts_test_module_analytical_characterization
  SOURCES
    ${ICTS_TEST}/module/analytical_characterization/AnalyticalFitTest.cc
    ${ICTS_TEST}/module/analytical_characterization/AnalyticalCharacterizationTest.cc
  LIBS
    icts_source_module_analytical_characterization
)
