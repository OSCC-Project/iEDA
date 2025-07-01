if(DEBUG_IRT_MONITOR)
    message(STATUS "RT: DEBUG_IRT_MONITOR")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_MONITOR")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_monitor
    ${IRT_TOOLKIT}/monitor/Monitor.cpp
)

target_link_libraries(irt_monitor
    PUBLIC
    irt_data_manager
)

target_include_directories(irt_monitor
    PUBLIC
    ${IRT_TOOLKIT}/monitor
)
