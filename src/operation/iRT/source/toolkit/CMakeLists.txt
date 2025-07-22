add_subdirectory(${IRT_TOOLKIT}/logger)
add_subdirectory(${IRT_TOOLKIT}/monitor)
add_subdirectory(${IRT_TOOLKIT}/utility)

add_library(irt_toolkit INTERFACE)

target_link_libraries(irt_toolkit
    INTERFACE
    irt_logger
    irt_monitor
    irt_utility
    irt_interface
)

target_include_directories(irt_toolkit
    INTERFACE
    ${IRT_TOOLKIT}
)
