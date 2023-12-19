if(NOT DEFINED CMD_BUILD)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/contest/IncTO_2023/scripts)
endif()

add_executable(IncTO_2023
    incto_2023.cpp
)

target_link_libraries(IncTO_2023 flow)
