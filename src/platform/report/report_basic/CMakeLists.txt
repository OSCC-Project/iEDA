add_library(report_basic
  report_basic.cpp
)

target_link_libraries(report_basic 
    PUBLIC
        report_table
        log
        time
        idm
        tool_manager
        def_service
)

target_include_directories(report_basic 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_UTILITY}/time
)

# SET(CMAKE_BUILD_TYPE "Debug")