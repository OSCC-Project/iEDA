add_library(icts_api_external_libs INTERFACE)

target_link_libraries(icts_api_external_libs INTERFACE idm ista-engine)

target_include_directories(
  icts_api_external_libs
  INTERFACE ${HOME_PLATFORM}/data_manager
            ${HOME_PLATFORM}/data_manager/file_manager ${HOME_ISTA}/api)
