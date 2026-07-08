add_library(icts_api_external_libs INTERFACE)

target_link_libraries(icts_api_external_libs INTERFACE
  idm
  log
  usage
  feature_db
)
