add_library(
  icts_test_common_dataset
  STATIC
  ${ICTS_TEST}/common/dataset/TestDataset.cc
)

target_link_libraries(
  icts_test_common_dataset
  PUBLIC
  icts_source_database_spatial
  icts_test_base
)
