cmake_minimum_required(VERSION 3.11)
SET (CMAKE_CXX_STANDARD 20)

AUX_SOURCE_DIRECTORY(./ SRC)

SET(LINK_unwind "unwind")

find_package(glog REQUIRED)
message(STATUS "Detected glog version: ${glog_VERSION}")

if(glog_VERSION VERSION_LESS_EQUAL "0.5.0")
    # For glog versions 0.5.0 and earlier, the signature of SignalHandle is (const char*, int)
    # We define GOOGLE_GLOG_VERSION to 50 to indicate a version like 0.5.0
    add_compile_definitions(GOOGLE_GLOG_VERSION=50)
else()
    # For glog 0.6.0 and later versions, the signature of SignalHandle is (const char*, size_t)
    # We define GOOGLE_GLOG_VERSION to 60 to indicate a version like 0.6.0
    add_compile_definitions(GOOGLE_GLOG_VERSION=60)
endif()

# Define GLOG_USE_GLOG_EXPORT for glog 0.7.1+ compatibility
add_definitions(-DGLOG_USE_GLOG_EXPORT)

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

target_include_directories(log
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)