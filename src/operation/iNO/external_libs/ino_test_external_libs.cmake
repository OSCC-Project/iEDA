add_library(ino_test_external_libs INTERFACE)

target_link_libraries(ino_test_external_libs
    INTERFACE
        gtest
        gtest_main
        pthread
)