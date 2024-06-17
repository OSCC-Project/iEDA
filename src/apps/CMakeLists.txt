# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_BUILD_TYPE "Debug")

if(NOT DEFINED CMD_BUILD)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/design/sky130_gcd)
endif()

# add GIT_VERSION to definitions
find_package(Git)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=40 --always OUTPUT_VARIABLE GIT_VERSION)
    string(STRIP "${GIT_VERSION}" iEDA_GIT_VERSION)
    message(STATUS "Git Version: " ${iEDA_GIT_VERSION})
else()
    message(WARNING "Git Version: not available")
    set(iEDA_GIT_VERSION "not available")
endif()

add_definitions(-DiEDA_GIT_VERSION="${iEDA_GIT_VERSION}")

add_executable(iEDA
    ieda_main.cpp
)

target_link_libraries(iEDA flow)
