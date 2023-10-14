add_library(contest_db
    contest_db.cpp
)

target_include_directories(contest_db 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)