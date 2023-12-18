add_subdirectory(${ICTS_SOLVER}/tools/tree_builder/bound_skew_tree)
add_subdirectory(${ICTS_SOLVER}/tools/tree_builder/concurrent_bst_salt)
add_subdirectory(${ICTS_SOLVER}/tools/tree_builder/local_legalization)

if(DEBUG_ICTS_TREE_BUILDER)
  message(STATUS "CTS: DEBUG_ICTS_TREE_BUILDER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_TREE_BUILDER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_tree_builder ${ICTS_SOLVER}/tools/tree_builder/TreeBuilder.cc)

target_link_libraries(
  icts_tree_builder
  PUBLIC icts_data_manager
         icts_api
         icts_database
         icts_bst
         icts_cbs
         icts_local_legalization
         icts_timing_propagator
         salt)

target_include_directories(
  icts_tree_builder
  PUBLIC ${ICTS_SOLVER}/database
         ${ICTS_SOLVER}/tools/tree_builder/bound_skew_tree
         ${ICTS_SOLVER}/tools/tree_builder/local_legalization
         ${ICTS_SOLVER}/tools/tree_builder/concurrent_bst_salt
         ${ICTS_SOLVER}/tools/timing_propagator
         src/third_party/salt)
