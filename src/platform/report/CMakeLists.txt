include(${HOME_CMAKE}/operation/idb.cmake)

add_subdirectory(report_basic)
add_subdirectory(report_db)
add_subdirectory(report_evaluator)
add_subdirectory(report_route)
add_subdirectory(report_place)
add_subdirectory(report_drc)

add_library(ieda_report
  report_manager.cpp
)

target_link_libraries(ieda_report 
    PRIVATE
        report_basic 
        ieda_report_db
        ieda_report_evaluator
        ieda_report_route
        ieda_report_place
        ieda_report_drc
)

target_include_directories(ieda_report 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
