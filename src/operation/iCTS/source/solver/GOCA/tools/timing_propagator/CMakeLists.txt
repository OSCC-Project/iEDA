if(DEBUG_ICTS_GOCA_TIMING_PROPAGATOR)
  message(STATUS "CTS: DEBUG_ICTS_GOCA_TIMING_PROPAGATOR")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_GOCA_TIMING_PROPAGATOR")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_goca_timing_propagator
            ${ICTS_SOLVER}/GOCA/tools/timing_propagator/TimingPropagator.cc)

target_link_libraries(icts_goca_timing_propagator
                      PUBLIC icts_data_manager icts_api icts_goca_database)

target_include_directories(icts_goca_timing_propagator
                           PUBLIC ${ICTS_SOLVER}/GOCA/database)
