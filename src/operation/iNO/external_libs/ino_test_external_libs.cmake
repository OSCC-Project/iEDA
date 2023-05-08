add_library(ino_test_external_libs INTERFACE)

target_link_libraries(ino_test_external_libs 
    INTERFACE
        libgtest.a 
        libgtest_main.a 
        pthread 
)