add_subdirectory(method)

add_library(solver_legalization
    LGMethodCreator.cc
)

target_link_libraries(solver_legalization 
    PRIVATE
        solver_lg_method
)

target_include_directories(solver_legalization 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/method
)

message("-- Solver-Legalization: ENABLE")