add_library(
  icts_test_common_data_pin_factory
  STATIC
  ${ICTS_TEST}/common/data/pin_factory/PinFactory.cc
)

target_link_libraries(
  icts_test_common_data_pin_factory
  PUBLIC
  icts_test_common_dataset
)
