# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_BUILD_TYPE "Debug")

if(NOT DEFINED CMD_BUILD)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /data/project_share/benchmark/t28/asic_top)
    # set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /data/project_share/benchmark/smic180)
endif()

add_executable(iEDA
    ieda_main.cpp
)

target_link_libraries(iEDA flow)
