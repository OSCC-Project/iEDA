# SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

option(TEST_IDB "If ON, test db." ON)
if(TEST_IDB)
    add_executable(test_idb  main.cpp)
    target_link_libraries(test_idb idb IdbBuilder def_builder def_service lef_builder lef_service )


    target_include_directories(test_idb 
        PUBLIC 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${HOME_DATABASE}/data/design
            ${HOME_DATABASE}/data/design/db_design
            ${HOME_DATABASE}/data/design/db_layout
            ${HOME_DATABASE}/manager/builder
            ${HOME_DATABASE}/manager/builder/def_builder
            ${HOME_DATABASE}/manager/builder/lef_builder
            ${HOME_DATABASE}/manager/service/def_service
            ${HOME_DATABASE}/manager/service/lef_service
    )

    # # add_executable(iDB_verilog ${CMAKE_CURRENT_SOURCE_DIR}/example/test_verilog.cpp)
    # target_link_libraries(iDB_verilog idb IdbBuilder data_process data_service def def_read def_write def_service lef lef_read lef_service )
endif()