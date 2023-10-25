# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_BUILD_TYPE "Debug")

if(NOT DEFINED CMD_BUILD)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/huangzengrong/oscc/iEDA/scripts/28nm/t28_gcd)
endif()

add_executable(iEDA
    ieda_main.cpp
)

target_link_libraries(iEDA flow)
