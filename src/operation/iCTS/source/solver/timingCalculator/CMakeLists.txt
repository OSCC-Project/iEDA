if(DEBUG_ICTS_TIMING_CALCULATOR)
  message(STATUS "CTS: DEBUG_ICTS_TIMING_CALCULATOR")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_TIMING_CALCULATOR")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_timing_calculator
            ${ICTS_SOLVER}/timingCalculator/TimingCalculator.cc)

target_link_libraries(icts_timing_calculator PUBLIC icts_data_manager icts_api)

target_include_directories(icts_timing_calculator
                           PUBLIC ${ICTS_SOLVER}/timingCalculator)
