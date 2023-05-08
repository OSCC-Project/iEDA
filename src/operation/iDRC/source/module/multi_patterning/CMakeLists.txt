add_library(idrc_multi_pattern
    MultiPatterning.cpp
    odd_cycle_finder/OddCycleFinder.cpp
    connected_component_finder/ConnectedComponentFinder.cpp
    colorable_checker/ColorableChecker.cpp

)

target_include_directories(idrc_multi_pattern
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/odd_cycle_finder
    ${CMAKE_CURRENT_SOURCE_DIR}/connected_component_finder
    ${CMAKE_CURRENT_SOURCE_DIR}/colorable_checker
)

target_link_libraries(idrc_multi_pattern
    PRIVATE
    idrc_db

)