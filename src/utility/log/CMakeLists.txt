cmake_minimum_required(VERSION 3.11)
SET (CMAKE_CXX_STANDARD 20)

AUX_SOURCE_DIRECTORY(./ SRC)

SET(LINK_glog   "glog")
SET(LINK_gflags "gflags")
SET(LINK_unwind "unwind")

if(BUILD_STATIC_LIB)
  ADD_LIBRARY(log ${SRC})
  SET(LINK_glog   "libglog.a")
  SET(LINK_gflags "libgflags.a")
  find_library(UNWIND_STATIC_LIB 
    NAMES libunwind.a
    PATHS /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
  )
  find_library(LZMA_STATIC_LIB
    NAMES liblzma.a
    PATHS /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
  )
  if(UNWIND_STATIC_LIB)
    SET(LINK_unwind ${UNWIND_STATIC_LIB} ${LZMA_STATIC_LIB} pthread)
  endif()
else()
  ADD_LIBRARY(log SHARED ${SRC})
endif()

TARGET_LINK_LIBRARIES(log PUBLIC ${LINK_glog} ${LINK_gflags} ${LINK_unwind})

target_include_directories(log
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)