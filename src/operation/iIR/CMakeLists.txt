cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")

aux_source_directory(./ SRC)
add_executable(iIR ${SRC})

set(RUST_PROJECT_NAME iir)
set(RUST_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/iir-rust/iir)
set(RUST_LIB_TYPE a)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/debug/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "")
else()
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/release/lib${RUST_PROJECT_NAME}.${RUST_LIB_TYPE})
    set(RUST_BUILD_CMD_OPTION "--release")
endif()

message("iIR rust lib path ${RUST_LIB_PATH}")

ADD_EXTERNAL_PROJ(iir)

target_link_libraries(iIR PRIVATE ${RUST_LIB_PATH})