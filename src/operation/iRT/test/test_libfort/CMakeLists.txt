add_executable(test_libfort
    ${IRT_TEST}/test_libfort/test_libfort.cpp
    ${IRT_TEST}/test_libfort/fort.c
)


target_include_directories(test_libfort
    PUBLIC
        ${IRT_TEST}/test_libfort
)