# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_BUILD_TYPE "Debug")

if(NOT DEFINED CMD_BUILD)
    # set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/design/sky130_gcd)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/28nm/t28)
endif()

add_executable(iEDA
    ieda_main.cpp
)

target_link_libraries(iEDA flow)
