add_library(ipl-test_external_libs INTERFACE)

target_link_libraries(ipl-test_external_libs
    INTERFACE
        libgtest.a
        libgtest_main.a
        pthread
)