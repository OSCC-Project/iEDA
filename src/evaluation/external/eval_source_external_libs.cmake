add_library(eval_source_external_libs INTERFACE)

target_link_libraries(eval_source_external_libs
    INTERFACE
        # database
        geometry_db
        idb
        IdbBuilder
        def_builder
        lef_builder
        def_service
        lef_service
        # utility
        usage
        # sta
        eval_ista_libs
)

target_include_directories(eval_source_external_libs
    INTERFACE
        # database
        ${HOME_DATABASE}/basic/geometry
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_DATABASE}/manager/builder
        ${HOME_DATABASE}/manager/builder/def_builder
        ${HOME_DATABASE}/manager/builder/lef_builder
        ${HOME_DATABASE}/manager/service/def_service
        ${HOME_DATABASE}/manager/service/lef_service
        # utility
        ${HOME_UTILITY}
)

target_include_directories(eval_source_external_libs
    SYSTEM INTERFACE
        # third_party
        ${HOME_THIRDPARTY}
        ${HOME_THIRDPARTY}/flute3
        ${HOME_THIRDPARTY}/magic_enum
)

