INCLUDE_DIRECTORIES(
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

LINK_LIBRARIES(
    geometry_db
    
    idb
    IdbBuilder
    def_builder
    lef_builder
    def_service
    lef_service
)