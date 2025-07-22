# model
if(DEBUG_ICTS_MODEL)
  message(STATUS "CTS: DEBUG_ICTS_MODEL")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_MODEL")
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
add_library(icts_model ${ICTS_MODEL}/ModelFactory.cc)

find_package(Eigen3 QUIET REQUIRED)
message(STATUS "CTS: Eigen3 ${EIGEN3_INCLUDE_DIR}")

target_link_libraries(icts_model PUBLIC Eigen3::Eigen)
target_include_directories(icts_model PUBLIC ${ICTS_MODEL})

