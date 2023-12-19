add_library(contest_process
    contest_process.cpp
)

target_link_libraries(contest_process contest_dm)

target_include_directories(contest_process 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)