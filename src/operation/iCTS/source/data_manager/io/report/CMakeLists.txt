if(DEBUG_ICTS_REPORT)
  message(STATUS "CTS: DEBUG_ICTS_REPORT")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_REPORT")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_report ${ICTS_DATA_MANAGER}/io/report/CtsReport.cc)

target_link_libraries(icts_report PUBLIC report_table)

target_include_directories(icts_report PUBLIC ${ICTS_DATA_MANAGER}/io)
