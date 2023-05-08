add_library(ieda_report_db
  report_db.cpp
  report_design.cpp
  report_layout.cpp
)

target_link_libraries(ieda_report_db 
    PUBLIC
        report_basic
        idm  
        tool_manager 
        def_service
        log
)

target_include_directories(ieda_report_db 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
