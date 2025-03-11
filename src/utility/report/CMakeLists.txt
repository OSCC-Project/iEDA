cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
if(BUILD_STATIC_LIB)
  add_library(report_table ${SRC})
else()
  add_library(report_table SHARED ${SRC})
endif()

target_link_libraries(report_table fort)

target_include_directories(report_table 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_THIRDPARTY}
)