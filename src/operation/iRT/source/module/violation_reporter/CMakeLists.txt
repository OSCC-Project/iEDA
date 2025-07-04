if(DEBUG_IRT_VIOLATION_REPORTER)
    message(STATUS "RT: DEBUG_IRT_VIOLATION_REPORTER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_VIOLATION_REPORTER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_violation_reporter
    ## module violation_reporter
    ${IRT_MODULE}/violation_reporter/ViolationReporter.cpp
)

target_link_libraries(irt_violation_reporter
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_violation_reporter
    PUBLIC
    ${IRT_MODULE}/violation_reporter/vr_data_manager
    ${IRT_MODULE}/violation_reporter
)
