cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")

aux_source_directory(./ SRC)
add_library(spef ${SRC})

set(RUST_PROJECT_NAME spef_parser)
set(RUST_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spef-parser)
set(RUST_LIB_TYPE a)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/debug/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "")
else()
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/release/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "--release")
endif()

message(STATUS "spef parser rust lib path ${RUST_LIB_PATH}")

ADD_EXTERNAL_PROJ(spef)

target_link_libraries(spef log ${RUST_LIB_PATH} dl)
target_include_directories(spef PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_spef ${CMAKE_CURRENT_SOURCE_DIR}/spef-parser/test/test.cpp)
target_link_libraries(test_spef spef)
target_include_directories(test_spef PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR})


