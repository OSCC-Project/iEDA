add_library(eval_idb_libs INTERFACE)
target_link_libraries(eval_idb_libs
    INTERFACE
        geometry_db

        idb
        IdbBuilder
        def_builder
        lef_builder
        def_service
        lef_service
)

target_include_directories(eval_idb_libs
    INTERFACE
        ##basic
        ${HOME_DATABASE}/basic/geometry
        ## iDB
        ${HOME_DATABASE}/tool_db/design
        ${HOME_DATABASE}/tool_db/design/db_design
        ${HOME_DATABASE}/tool_db/design/db_layout
        ##builder
        ${HOME_DATABASE}/manager/builder
        ${HOME_DATABASE}/manager/builder/def_builder
        ${HOME_DATABASE}/manager/builder/lef_builder
        ##service
        ${HOME_DATABASE}/manager/service/def_service
        ${HOME_DATABASE}/manager/service/lef_service
)
