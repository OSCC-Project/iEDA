add_library(icts_source_database_qor INTERFACE)

target_include_directories(
  icts_source_database_qor
  INTERFACE
  ${ICTS_DATABASE_QOR}
  ${ICTS_DATABASE}
)
