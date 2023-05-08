aux_source_directory(. REPORT_ROUTE_SRC)


add_library(ieda_report_route ${REPORT_ROUTE_SRC})

target_link_libraries(ieda_report_route 
    PUBLIC
        report_basic
        idm  
        tool_manager 
        def_service
        log
)

target_include_directories(ieda_report_route 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)