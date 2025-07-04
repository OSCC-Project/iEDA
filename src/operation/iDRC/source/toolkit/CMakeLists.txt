add_subdirectory(${IDRC_TOOLKIT}/logger)
add_subdirectory(${IDRC_TOOLKIT}/monitor)
add_subdirectory(${IDRC_TOOLKIT}/utility)

add_library(idrc_toolkit INTERFACE)

target_link_libraries(idrc_toolkit
    INTERFACE
    idrc_logger
    idrc_monitor
    idrc_utility
    idrc_interface
)

target_include_directories(idrc_toolkit
    INTERFACE
    ${IDRC_TOOLKIT}
)
