aux_source_directory(. REPORT_PLACE_SRC)


add_library(ieda_report_place ${REPORT_PLACE_SRC})

target_link_libraries(ieda_report_place 
    PUBLIC
        report_basic
        idm  
        tool_manager 
        def_service
        log
        ipl-api
        ipl-source
        ipl-api_external_libs
)

target_include_directories(ieda_report_place 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
