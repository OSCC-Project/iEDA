if(DEBUG_IRT_DATA_MANAGER)
    message(STATUS "RT: DEBUG_IRT_DATA_MANAGER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_DATA_MANAGER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_data_manager
    ${IRT_DATA_MANAGER}/DataManager.cpp
)

target_link_libraries(irt_data_manager
    PUBLIC
    irt_toolkit
)

target_include_directories(irt_data_manager
    PUBLIC
    ${IRT_DATA_MANAGER}/advance
    ${IRT_DATA_MANAGER}/basic
    ${IRT_DATA_MANAGER}
)
