set(CMAKE_BUILD_TYPE "Debug")

add_executable(test_run_to ${ITO_TEST_PATH}/run_to.cpp)
target_link_libraries(test_run_to 
    PUBLIC
        ito_source
)
