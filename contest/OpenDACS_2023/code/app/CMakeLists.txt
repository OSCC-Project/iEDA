# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_BUILD_TYPE "Debug")

if(NOT DEFINED CMD_BUILD)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/contest/OpenDACS_2023/scripts)
endif()

add_executable(iEDA_contest
    ieda_contest.cpp
)

target_link_libraries(iEDA_contest flow)
