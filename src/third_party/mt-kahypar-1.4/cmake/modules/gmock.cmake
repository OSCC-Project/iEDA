# taken from http://johnlamp.net/cmake-tutorial-5-functionally-improved-testing.html
function(add_gmock_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

    set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)

    add_test(${target} ${target})

    add_custom_command(TARGET ${target}
                       POST_BUILD
                       COMMAND cp ${target} ${target}_failed
                       COMMAND ${target}
                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                       COMMENT "Running ${target}" VERBATIM)

    add_custom_command(TARGET ${target}
                       POST_BUILD
                       COMMAND rm -f ${target}_failed
                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                       COMMENT "Running ${target}" VERBATIM)
endfunction()
