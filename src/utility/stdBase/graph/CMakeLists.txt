cmake_minimum_required(VERSION 3.11)
SET (CMAKE_CXX_STANDARD 17)

AUX_SOURCE_DIRECTORY(./ SRC)
if(BUILD_STATIC_LIB)
  ADD_LIBRARY(graph STATIC ${SRC})
else()
  ADD_LIBRARY(graph SHARED ${SRC})
endif()

target_link_libraries(graph PUBLIC absl::btree)
