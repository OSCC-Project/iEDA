add_library(icts_source_external_libs INTERFACE)

target_link_libraries(
  icts_source_external_libs
  INTERFACE geometry_db
            idb
            IdbBuilder
            def_builder
            lef_builder
            def_service
            lef_service)

target_include_directories(
  icts_source_external_libs
  INTERFACE ${HOME_DATABASE}/db ${HOME_DATABASE}/builder/builder
            ${HOME_DATABASE}/builder/def_builder/def_service
            ${HOME_DATABASE}/builder/lef_builder/lef_service)
