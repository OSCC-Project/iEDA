add_library(contest_preprocess
    contest_preprocess.cpp
    placer.cpp
    router.cpp
)

target_link_libraries(contest_preprocess contest_dm)

target_include_directories(contest_preprocess 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/astar
        ${CMAKE_CURRENT_SOURCE_DIR}/gridmap
)