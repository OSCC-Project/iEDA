add_library(eval_test_external_libs INTERFACE)

target_link_libraries(eval_test_external_libs 
    INTERFACE
        libgtest.a
        libgtest_main.a 
        pthread       
)
