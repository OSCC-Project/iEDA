add_library(ito_test_external_libs INTERFACE)

target_link_libraries(ito_test_external_libs 
    INTERFACE
        libgtest.a 
        libgtest_main.a 
        pthread 
)