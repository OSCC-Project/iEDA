if(DEBUG_ICTS_HCTS)
  message(STATUS "CTS: DEBUG_ICTS_HCTS")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_HCTS")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_hcts ${ICTS_SOLVER}/HCTS/HCTS.cc)

target_link_libraries(icts_hcts PUBLIC icts_data_manager icts_clustering
                                       icts_api)

target_include_directories(icts_hcts PUBLIC ${ICTS_SOLVER}/HCTS
                                            ${ICTS_DATABASE})
