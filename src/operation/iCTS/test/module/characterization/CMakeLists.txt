add_subdirectory(fixture)

icts_add_test_executable(
  icts_test_module_characterization
  PUBLIC_RUNTIME
  SOURCES
    ${ICTS_TEST}/module/characterization/BufferingPatternTest.cc
    ${ICTS_TEST}/module/characterization/SegmentJoinTest.cc
    ${ICTS_TEST}/module/characterization/HTreeJoinTest.cc
    ${ICTS_TEST}/module/characterization/PrunerTest.cc
)

if(ICTS_BUILD_REALTECH_TESTS)
  icts_add_test_executable(
    icts_test_module_characterization_realtech
    REALTECH
    SOURCES
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechSmokeTest.cc
    LIBS
      icts_test_module_characterization_fixture
      icts_test_common_realtech_setup
      icts_test_common_io
  )

  target_compile_definitions(
    icts_test_module_characterization_realtech
    PRIVATE
      ICTS_ENABLE_SLOW_REALTECH_REGRESSION=0
  )
endif()

if(ICTS_BUILD_REALTECH_TESTS AND ICTS_BUILD_SLOW_REALTECH_TESTS)
  icts_add_test_executable(
    icts_test_module_characterization_realtech_regression
    REALTECH
    SOURCES
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechSmokeTest.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechComposeGap.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechExactRegressionTest.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechSurfaceFit.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechLimitResolutionTest.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechFunctionComposeRegressionTest.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechFunctionGap.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechFunctionModel.cc
      ${ICTS_TEST}/module/characterization/CharacterizationRealTechPowerAccountingTest.cc
    LIBS
      icts_test_module_characterization_fixture
      icts_test_common_realtech_setup
      icts_test_common_io
  )

  target_compile_definitions(
    icts_test_module_characterization_realtech_regression
    PRIVATE
      ICTS_ENABLE_SLOW_REALTECH_REGRESSION=1
  )
endif()
