add_library(
  icts_test_common_io
  STATIC
  ${ICTS_TEST}/common/io/TestArtifactIO.cc
)

target_link_libraries(
  icts_test_common_io
  PRIVATE
  log
  icts_source_utils_logger
  icts_test_common_dataset
)

icts_add_test_executable(
  icts_test_common_io_artifact
  SOURCES
    ${ICTS_TEST}/common/io/TestArtifactIOTest.cc
  LIBS
    icts_test_common_io
)
