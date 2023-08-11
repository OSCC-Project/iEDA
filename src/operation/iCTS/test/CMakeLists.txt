if(DEBUG_ICTS_TEST)
  message(STATUS "CTS: DEBUG_ICTS_TEST")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_TEST")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(PY_MODEL)
  add_executable(icts_py_test ${ICTS_TEST}/PyTest.cc)
  target_link_libraries(icts_py_test PUBLIC icts_source icts_test_external_libs)
endif()

add_executable(icts_solver_test ${ICTS_TEST}/SolverTest.cc)
target_link_libraries(icts_solver_test PUBLIC icts_source
                                              icts_api
                                              icts_test_external_libs)
