if(DEBUG_ICTS_DATABASE)
  message(STATUS "CTS: DEBUG_ICTS_DATABASE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_DATABASE")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(
  icts_database
  ${ICTS_DATA_MANAGER}/database/CtsDesign.cc
  ${ICTS_DATA_MANAGER}/database/CtsInstance.cc
  ${ICTS_DATA_MANAGER}/database/CtsNet.cc
  ${ICTS_DATA_MANAGER}/database/CtsPin.cc)

target_link_libraries(icts_database PUBLIC icts_data_manager icts_model)

target_include_directories(icts_database PUBLIC ${ICTS_DATA_MANAGER}/config
                                                ${ICTS_MODEL})
