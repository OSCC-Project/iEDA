if(DEBUG_ICTS_CBS)
  message(STATUS "CTS: DEBUG_ICTS_CBS")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_CBS")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_cbs
            ${ICTS_SOLVER}/tools/tree_builder/concurrent_bst_salt/CBS.cc)

target_link_libraries(icts_cbs PUBLIC icts_database icts_api
                                      icts_timing_propagator salt)

target_include_directories(
  icts_cbs
  PUBLIC ${ICTS_SOLVER}/tools/tree_builder/concurrent_bst_salt
         ${ICTS_SOLVER}/tools/timing_propagator
         ${PROJECT_SOURCE_DIR}/src/third_party/salt)
