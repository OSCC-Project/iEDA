add_library(irt_source_external_libs INTERFACE)

target_link_libraries(irt_source_external_libs 
    INTERFACE
        geometry_db
        idb
        IdbBuilder
        def_builder
        lef_builder
        def_service
        lef_service
        ls_assigner
)

target_include_directories(irt_source_external_libs
    INTERFACE
        ${HOME_DATABASE}/db
        ${HOME_DATABASE}/builder/builder
        ${HOME_DATABASE}/builder/def_builder/def_service
        ${HOME_DATABASE}/builder/lef_builder/lef_service
        ${HOME_UTILITY}
)
