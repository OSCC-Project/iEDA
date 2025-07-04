if(DEBUG_IDRC_DATA_MANAGER)
    message(STATUS "DRC: DEBUG_IDRC_DATA_MANAGER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_DATA_MANAGER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_data_manager
    ${IDRC_DATA_MANAGER}/DataManager.cpp
)

target_link_libraries(idrc_data_manager
    PUBLIC
    idrc_toolkit
)

target_include_directories(idrc_data_manager
    PUBLIC
    ${IDRC_DATA_MANAGER}/advance
    ${IDRC_DATA_MANAGER}/basic
    ${IDRC_DATA_MANAGER}/design_rule
    ${IDRC_DATA_MANAGER}
)
