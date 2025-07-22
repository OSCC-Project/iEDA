if(DEBUG_IDRC_UTILITY)
    message(STATUS "DRC: DEBUG_IDRC_UTILITY")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_UTILITY")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_utility
    ${IDRC_TOOLKIT}/utility/Utility.cpp
)

target_link_libraries(idrc_utility
    PUBLIC
    idrc_data_manager
)

target_include_directories(idrc_utility
    PUBLIC
    ${IDRC_TOOLKIT}/utility
)
