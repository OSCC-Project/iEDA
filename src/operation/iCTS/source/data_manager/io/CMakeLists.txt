if(DEBUG_ICTS_IO)
  message(STATUS "CTS: DEBUG_ICTS_IO")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_IO")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(report)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_THIRDPARTY}/json
)

add_library(icts_io ${ICTS_DATA_MANAGER}/io/CtsDBWrapper.cc
                    ${ICTS_DATA_MANAGER}/io/GDSPloter.cc)

target_link_libraries(icts_io PUBLIC icts_data_manager icts_report)

target_include_directories(icts_io PUBLIC ${ICTS_DATA_MANAGER}/io)
