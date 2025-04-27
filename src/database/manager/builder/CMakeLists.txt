# add_subdirectory(data_builder)
add_subdirectory(def_builder)
add_subdirectory(lef_builder)
add_subdirectory(verilog_builder)
add_subdirectory(gds_builder)
add_subdirectory(json_builder)

if(BUILD_STATIC_LIB)
    add_library(IdbBuilder
        builder.cpp
        buildNet.cpp
        buildBus.cpp
        buildLefData.cpp
    )
else()
    add_library(IdbBuilder SHARED
        builder.cpp
        buildNet.cpp
        buildBus.cpp
        buildLefData.cpp
    )

endif()

target_link_libraries(IdbBuilder def_service def_builder lef_service lef_builder verilog_builder gds_builder json_builder)

target_include_directories(IdbBuilder 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/def_builder
        ${CMAKE_CURRENT_SOURCE_DIR}/lef_builder
        ${CMAKE_CURRENT_SOURCE_DIR}/verilog_builder
        ${CMAKE_CURRENT_SOURCE_DIR}/json_builder
        ${CMAKE_CURRENT_SOURCE_DIR}/gds_builder
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_DATABASE}/manager/service/def_service
        ${HOME_DATABASE}/manager/service/lef_service
)
