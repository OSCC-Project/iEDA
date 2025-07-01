if(DEBUG_IDRC_INTERFACE)
    message(STATUS "DRC: DEBUG_IDRC_INTERFACE")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_INTERFACE")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_interface
    ${IDRC_INTERFACE}/DRCInterface.cpp
)

target_link_libraries(idrc_interface
    PRIVATE
    idrc_source
    idm
)

target_include_directories(idrc_interface
    PUBLIC
    ${IDRC_INTERFACE}
)
