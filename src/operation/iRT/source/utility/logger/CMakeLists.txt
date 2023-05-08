if(DEBUG_IRT_LOGGER)            
    message(STATUS "RT: DEBUG_IRT_LOGGER")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_LOGGER")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_logger
    ${IRT_UTILITY}/logger/Logger.cpp
)

target_link_libraries(irt_logger 
    PUBLIC
        irt_data_manager
)

target_include_directories(irt_logger 
    PUBLIC
        ${IRT_UTILITY}/logger
)
