add_library(
  icts_test_common_realtech_asset_loader
  STATIC
  ${ICTS_TEST}/common/realtech/asset/RealTechAssetLoader.cc
  ${ICTS_TEST}/common/realtech/asset/RealTechPinCapProbe.cc
)

target_link_libraries(
  icts_test_common_realtech_asset_loader
  PRIVATE
  icts_test_common_dataset
  icts_test_common_io
  icts_test_realtech_base
)
