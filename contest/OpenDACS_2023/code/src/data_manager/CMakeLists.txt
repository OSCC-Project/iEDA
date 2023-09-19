add_subdirectory(builder)

add_library(contest_dm
    contest_dm.cpp
)

target_link_libraries(contest_dm contest_builder contest_db)

target_include_directories(contest_dm 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)