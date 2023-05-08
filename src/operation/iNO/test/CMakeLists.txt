set(CMAKE_BUILD_TYPE "Debug")

add_executable(test_run_no ${INO_TEST_PATH}/run_no.cpp)
target_link_libraries(test_run_no 
    PUBLIC
        ino_source 
        # irt_test_external_libs 
)
