add_subdirectory(pin_factory)
add_subdirectory(distribution)

add_library(
  icts_test_common_data
  STATIC
  ${ICTS_TEST}/common/data/TestDataGenerator.cc
)

target_link_libraries(
  icts_test_common_data
  PUBLIC
  icts_test_common_dataset
  PRIVATE
  icts_test_common_data_pin_factory
  icts_test_common_data_distribution
)
