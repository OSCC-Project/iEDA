CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET(CMAKE_CXX_STANDARD 20)

SET(CMAKE_BUILD_TYPE "Release")

AUX_SOURCE_DIRECTORY(./ SRC)

ADD_LIBRARY(aocv-parser STATIC ${SRC})

TARGET_LINK_LIBRARIES(aocv-parser PUBLIC absl::btree)
