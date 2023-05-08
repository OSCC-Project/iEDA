add_library(ito_source_external_libs INTERFACE)

target_link_libraries(ito_source_external_libs 
    INTERFACE
        geometry_db
        idb
        IdbBuilder
        def_builder
        lef_builder
        def_service
        lef_service
        usage
)

target_include_directories(ito_source_external_libs
    INTERFACE
        ${HOME_ISR}/module/flute3
        ${HOME_DATABASE}/db
        ${HOME_DATABASE}/builder/builder
        ${HOME_DATABASE}/builder/def_builder/def_service
        ${HOME_DATABASE}/builder/lef_builder/lef_service
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_UTILITY}
)
