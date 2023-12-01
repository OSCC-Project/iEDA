if(DEBUG_ICTS_EVALUATOR)
  message(STATUS "CTS: DEBUG_ICTS_EVALUATOR")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_EVALUATOR")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_evaluator ${ICTS_MODULE}/evaluator/Evaluator.cc)

target_link_libraries(icts_evaluator PUBLIC icts_data_manager icts_module
                                            icts_report)

target_include_directories(icts_evaluator PUBLIC ${ICTS_MODULE}/evaluator)
