add_executable(mt_kahypar_tests run_tests.cpp)
target_link_libraries(mt_kahypar_tests gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET mt_kahypar_tests PROPERTY CXX_STANDARD 17)
set_property(TARGET mt_kahypar_tests PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(mt_kahypar_tests ${Boost_LIBRARIES} TBB::tbb TBB::tbbmalloc_proxy)

add_subdirectory(datastructures)
add_subdirectory(interface)
add_subdirectory(io)
add_subdirectory(parallel)
add_subdirectory(partition)

set(PARTITIONING_SUITE_TARGETS ${PARTITIONING_SUITE_TARGETS} mt_kahypar_tests PARENT_SCOPE)
