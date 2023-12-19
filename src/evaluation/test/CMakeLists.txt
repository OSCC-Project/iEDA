# enable debug
# set(CMAKE_BUILD_TYPE "Debug")

# add_executable(evalTest WirelengthTest.cpp)
# add_executable(evalTest CongestionTest.cpp)
# add_executable(evalTest GDSWrapperTest.cpp)
# add_executable(evalTest GDSAPITest.cpp)
# add_executable(evalTest WirelengthAPITest.cpp)
add_executable(evalTest CongAPITest.cpp)

target_link_libraries(evalTest PUBLIC eval_api eval_source eval_source_external_libs)
target_link_libraries(evalTest PUBLIC gtest gtest_main report_table)
