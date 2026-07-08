add_library(
  icts_test_common_realtech_setup
  STATIC
  ${ICTS_TEST}/common/realtech/setup/RealTechDesignSetup.cc
)

target_link_libraries(
  icts_test_common_realtech_setup
  PUBLIC
  icts_test_common_dataset
  PRIVATE
  log
  icts_test_realtech_base
  icts_test_common_realtech_asset_loader
  icts_test_common_realtech_load_factory
)
