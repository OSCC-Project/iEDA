if(DEBUG_IDRC_MONITOR)
    message(STATUS "DRC: DEBUG_IDRC_MONITOR")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_MONITOR")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_monitor
    ${IDRC_TOOLKIT}/monitor/Monitor.cpp
)

target_link_libraries(idrc_monitor
    PUBLIC
    idrc_data_manager
)

target_include_directories(idrc_monitor
    PUBLIC
    ${IDRC_TOOLKIT}/monitor
)
