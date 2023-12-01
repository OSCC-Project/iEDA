cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")

aux_source_directory(./ SRC)

add_library(spef ${SRC})

include(ExternalProject)

set(RUST_PROJECT_NAME spef_parser)
set(RUST_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spef-parser)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/debug/lib${RUST_PROJECT_NAME}.so)
    set(RUST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/rust_output/debug)
    set(RUST_BUILD_CMD_OPTION "")
else()
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/release/lib${RUST_PROJECT_NAME}.so)
    set(RUST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/rust_output/release)
    set(RUST_BUILD_CMD_OPTION "--release")
endif()

ExternalProject_Add(
    ${RUST_PROJECT_NAME}
    PREFIX ${RUST_PROJECT_DIR}
    SOURCE_DIR ${RUST_PROJECT_DIR}
    BINARY_DIR ${RUST_PROJECT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cargo build ${RUST_BUILD_CMD_OPTION}
    INSTALL_COMMAND ""
    BUILD_ALWAYS 1
)

file(MAKE_DIRECTORY ${RUST_OUTPUT_DIR})
message("spef parser rust lib path ${RUST_LIB_PATH}")
message("spef parser rust outut path ${RUST_OUTPUT_DIR}")

add_custom_command(
    TARGET ${RUST_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${RUST_LIB_PATH}
            ${RUST_OUTPUT_DIR}
)

target_link_libraries(spef log ${RUST_LIB_PATH})
target_include_directories(spef PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_spef ${CMAKE_CURRENT_SOURCE_DIR}/spef-parser/test/test.cpp)
target_link_libraries(test_spef spef)
target_include_directories(test_spef PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR})


