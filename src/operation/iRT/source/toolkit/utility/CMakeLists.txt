if(DEBUG_IRT_UTILITY)
    message(STATUS "RT: DEBUG_IRT_UTILITY")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_UTILITY")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_utility
    ${IRT_TOOLKIT}/utility/Utility.cpp
)

target_link_libraries(irt_utility
    PUBLIC
    irt_data_manager
)

target_include_directories(irt_utility
    PUBLIC
    ${IRT_TOOLKIT}/utility
)
