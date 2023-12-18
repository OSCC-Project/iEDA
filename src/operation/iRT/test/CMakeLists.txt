# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")


add_subdirectory(${IRT_TEST}/process_benchmark)
add_subdirectory(${IRT_TEST}/process_guide)
add_subdirectory(${IRT_TEST}/test_boost)
add_subdirectory(${IRT_TEST}/test_libfort)
