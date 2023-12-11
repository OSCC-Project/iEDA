if(DEBUG_ICTS_SOLVER)
  message(STATUS "CTS: DEBUG_ICTS_SOLVER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_SOLVER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(${ICTS_SOLVER}/database)
add_subdirectory(${ICTS_SOLVER}/tools)

add_library(icts_solver ${ICTS_SOLVER}/Solver.cc)

target_link_libraries(icts_solver PUBLIC icts_database icts_api
                                         icts_solver_database icts_tools)

target_include_directories(
  icts_solver PUBLIC ${ICTS_SOLVER}/Solver ${ICTS_SOLVER}/database
                     ${ICTS_SOLVER}/tools)
