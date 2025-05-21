# find_package(Boost REQUIRED )

aux_source_directory(property_parser parser_src)
add_library(lef_builder
    lef_read.cpp
    ${parser_src}
)

target_include_directories(lef_builder 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_DATABASE}/basic/geometry
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_DATABASE}/manager/service/def_service
        ${HOME_DATABASE}/manager/service/lef_service
)

target_include_directories(lef_builder 
    SYSTEM PUBLIC 
        ${HOME_THIRDPARTY}/lefdef/lef
)

target_link_libraries(lef_builder PRIVATE lef geometry_db str)
target_link_libraries(lef_builder PUBLIC absl::strings)

option(TEST_LEFPARSER "If ON, test db." OFF)
if(TEST_LEFPARSER)
    find_package(GTest REQUIRED)
    add_executable(test_parser)
    aux_source_directory(property_parser/lef58_property/test testsrc)
    target_sources(test_parser PUBLIC ${testsrc})
    target_include_directories(test_parser PRIVATE         
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/manager/service/lef_service
    )
    target_link_libraries(test_parser libgtest.a libgtest_main.a pthread str)
endif()