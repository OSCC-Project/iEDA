cmake_minimum_required(VERSION 3.11)
SET (CMAKE_CXX_STANDARD 20)

AUX_SOURCE_DIRECTORY(./ SRC)

SET(LINK_unwind "unwind")

if(BUILD_STATIC_LIB)
  SET(LINK_glog   "libglog.a")
  SET(LINK_gflags "libgflags.a")
  ADD_LIBRARY(log STATIC ${SRC})
else()
  SET(LINK_glog   "glog")
  SET(LINK_gflags "gflags")
  ADD_LIBRARY(log SHARED ${SRC})
endif()

TARGET_LINK_LIBRARIES(log PUBLIC ${LINK_glog} ${LINK_gflags} ${LINK_unwind})
