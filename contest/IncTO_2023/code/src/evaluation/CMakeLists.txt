add_library(contest_evaluation
    contest_evaluation.cpp
)

target_link_libraries(contest_evaluation contest_dm)

target_include_directories(contest_evaluation 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)