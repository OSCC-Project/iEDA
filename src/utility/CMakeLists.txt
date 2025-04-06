cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

# add include and lib dirs
include_directories(SYSTEM ${HOME_THIRDPARTY})
include_directories(${HOME_UTILITY}/stdBase/include)
include_directories(${HOME_UTILITY}/stdBase/graph)
include_directories(${HOME_UTILITY}/log)
include_directories(${HOME_UTILITY}/string)
include_directories(${HOME_UTILITY}/tcl)
include_directories(${HOME_UTILITY})

link_directories(${CMAKE_BINARY_DIR}/lib)

add_subdirectory(json)
add_subdirectory(log)
add_subdirectory(string)
add_subdirectory(tcl)
add_subdirectory(time)
add_subdirectory(stdBase)
add_subdirectory(usage)
add_subdirectory(report)
add_subdirectory(memory)

option(BASE_RUN_TESTS "If ON, the tests will be run." OFF)

if(BASE_RUN_TESTS)
  message(STATUS "RUN BASE TESTS")

  # build test
  aux_source_directory(./test SourceFiles)
  add_executable(base_test ${SourceFiles})

  set(MyLibs
      log
      tcl
      str
      time
      graph)

  target_link_libraries(
    base_test
    gmock_main
    gtest
    gmock
    pthread
    ${MyLibs})

  add_custom_command(
    TARGET base_test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SO_FILES}
            ${CMAKE_CURRENT_BINARY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GLOG_SO_FILES}
            ${CMAKE_CURRENT_BINARY_DIR}/)

endif(BASE_RUN_TESTS)
