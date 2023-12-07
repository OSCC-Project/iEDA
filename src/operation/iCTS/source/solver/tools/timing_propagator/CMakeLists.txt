if(DEBUG_ICTS_TIMING_PROPAGATOR)
  message(STATUS "CTS: DEBUG_ICTS_TIMING_PROPAGATOR")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_TIMING_PROPAGATOR")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_timing_propagator
            ${ICTS_SOLVER}/tools/timing_propagator/TimingPropagator.cc)

target_link_libraries(icts_timing_propagator PUBLIC icts_data_manager icts_api
                                                    icts_solver_database)

target_include_directories(icts_timing_propagator
                           PUBLIC ${ICTS_SOLVER}/database)
