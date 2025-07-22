if(DEBUG_IDRC_LOGGER)
    message(STATUS "DRC: DEBUG_IDRC_LOGGER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_LOGGER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_logger
    ${IDRC_TOOLKIT}/logger/Logger.cpp
)

target_link_libraries(idrc_logger
    PUBLIC
    idrc_data_manager
)

target_include_directories(idrc_logger
    PUBLIC
    ${IDRC_TOOLKIT}/logger
)
