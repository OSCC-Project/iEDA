add_library(imp-idb-lib INTERFACE)
target_link_libraries(imp-idb-lib
    INTERFACE
    idb
    IdbBuilder
    def_builder
    lef_builder
    def_service
    lef_service
    )
target_include_directories(imp-idb-lib
    INTERFACE
    ${HOME_DATABASE}/db
    ${HOME_DATABASE}/builder/builder
    ${HOME_DATABASE}/builder/def_builder/def_service
    ${HOME_DATABASE}/builder/lef_builder/lef_service
    )