add_library(contest_builder
    contest_wrapper.cpp
    guide_parser.cpp
)

target_link_libraries(contest_builder idb idm contest_db)

target_include_directories(contest_builder 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)