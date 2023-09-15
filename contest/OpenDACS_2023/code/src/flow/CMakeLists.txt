add_library(contest_flow
    contest_flow.cpp
)

target_link_libraries(contest_flow contest_dm contest_process)

target_include_directories(contest_flow 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
