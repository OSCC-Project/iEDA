cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")

aux_source_directory(./ SRC)
add_library(liberty ${SRC})

set(RUST_PROJECT_NAME liberty_parser)
set(RUST_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib-rust/liberty-parser)
set(RUST_LIB_TYPE a)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/debug/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "")
else()
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/release/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "--release")
endif()

message(STATUS "liberty parser rust lib path ${RUST_LIB_PATH}")

ADD_EXTERNAL_PROJ(liberty)

target_link_libraries(liberty str sta-solver log ${RUST_LIB_PATH} dl)

target_include_directories(liberty PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${HOME_THIRDPARTY}/parser/liberty/)

add_executable(test_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib-rust/liberty-parser/test/test.cpp)
target_link_libraries(test_lib liberty)
target_include_directories(test_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS test_lib DESTINATION /home/taosimin/bin/iEDA_bin)