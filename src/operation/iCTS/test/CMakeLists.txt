if(DEBUG_ICTS_TEST)
  message(STATUS "CTS: DEBUG_ICTS_TEST")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_TEST")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# TreeBuilderTest
add_executable(icts_tree_builder_test ${ICTS_TEST}/TreeBuilderTest.cc)
target_link_libraries(
  icts_tree_builder_test PUBLIC icts_source icts_api icts_test_external_libs
                                gtest_main)
target_include_directories(icts_tree_builder_test PUBLIC ${ICTS_TEST})

# AnnealOptTest
add_executable(icts_anneal_opt_test ${ICTS_TEST}/AnnealOptTest.cc)
target_link_libraries(
  icts_anneal_opt_test PUBLIC icts_source icts_api icts_test_external_libs
                              gtest_main)
target_include_directories(icts_anneal_opt_test PUBLIC ${ICTS_TEST})
