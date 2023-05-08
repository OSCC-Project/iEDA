if(DEBUG_IRT_REPORTER)            
    message(STATUS "RT: DEBUG_IRT_REPORTER")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_REPORTER")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_reporter
    ${IRT_UTILITY}/reporter/Reporter.cpp
)

target_link_libraries(irt_reporter 
    PUBLIC
        irt_data_manager
)

target_include_directories(irt_reporter 
    PUBLIC
        ${IRT_UTILITY}/reporter
)
