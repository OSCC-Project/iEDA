add_library(
  icts_test_common_scoped_log_file
  STATIC
  ${ICTS_TEST}/common/logging/ScopedLogFile.cc
)

target_link_libraries(
  icts_test_common_scoped_log_file
  PRIVATE
  icts_test_common_io
  PUBLIC
  icts_test_base
)

icts_add_test_executable(
  icts_test_common_logging
  SOURCES
    ${ICTS_TEST}/common/logging/ScopedLogFileTest.cc
  LIBS
    icts_test_common_io
    icts_test_common_scoped_log_file
)
