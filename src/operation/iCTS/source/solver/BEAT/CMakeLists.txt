if(DEBUG_ICTS_BEAT)
  message(STATUS "CTS: DEBUG_ICTS_BEAT")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_BEAT")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_beat ${ICTS_SOLVER}/BEAT/BEAT.cc)

target_link_libraries(icts_beat PUBLIC icts_database icts_api
                                       icts_timing_propagator salt)

target_include_directories(
  icts_beat
  PUBLIC ${ICTS_SOLVER}/BEAT ${ICTS_SOLVER}/GOCA/tools/timing_propagator
         ${PROJECT_SOURCE_DIR}/src/third_party/salt)
