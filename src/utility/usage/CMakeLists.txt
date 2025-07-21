cmake_minimum_required(VERSION 3.11)
SET (CMAKE_CXX_STANDARD 20)

AUX_SOURCE_DIRECTORY(./ SRC)
if(BUILD_STATIC_LIB)
  ADD_LIBRARY(usage ${SRC})
else()
  ADD_LIBRARY(usage SHARED ${SRC})
endif()
target_include_directories(usage 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
