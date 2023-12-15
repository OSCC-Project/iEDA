if(DEBUG_ICTS_SOLVER_DATABASE)
  message(STATUS "CTS: DEBUG_ICTS_SOLVER_DATABASE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_SOLVER_DATABASE")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(
  icts_solver_database
  ${ICTS_SOLVER}/database/Pin.cc ${ICTS_SOLVER}/database/Inst.cc
  ${ICTS_SOLVER}/database/Net.cc)

target_link_libraries(icts_solver_database PUBLIC icts_api icts_data_manager)

target_include_directories(icts_solver_database PUBLIC ${ICTS_SOLVER}/database)
