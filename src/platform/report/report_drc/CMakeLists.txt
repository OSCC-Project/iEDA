add_library(ieda_report_drc
  report_drc.cpp
)

target_link_libraries(ieda_report_drc 
    PUBLIC
        report_basic
        tool_manager 
        def_service
        log
        idm
)

target_include_directories(ieda_report_drc 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
