add_library(tcl_report 
    tcl_report_db/tcl_report_db.cpp
    tcl_report_evl/tcl_report_evl.cpp
    tcl_report_pr/tcl_report_route.cpp
    tcl_report_pr/tcl_report_place.cpp
)

target_link_libraries(tcl_report 
    PUBLIC  
        tcl
        shell-cmd
        str
    PRIVATE
        ieda_report
)


target_include_directories(tcl_report 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tcl_report_db
    ${CMAKE_CURRENT_SOURCE_DIR}/tcl_report_evl
    ${CMAKE_CURRENT_SOURCE_DIR}/tcl_report_pr
)



