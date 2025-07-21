cmake_minimum_required(VERSION 3.11)
set (CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
if(BUILD_STATIC_LIB)
  add_library(str ${SRC})
else()
  add_library(str SHARED ${SRC})
endif()

target_link_libraries(str PUBLIC absl::strings)

target_include_directories(str
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
