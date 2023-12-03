cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNDEBUG")

################################################################
#
# Flex/bison scanner/parsers
#
################################################################

# find_package(FLEX)
# find_package(BISON)

# # Liberty scan/parse.
# flex_target(LibertyLex ${CMAKE_CURRENT_SOURCE_DIR}/LibertyLex.l ${CMAKE_CURRENT_SOURCE_DIR}/LibertyLex.cc)
# bison_target(LibertyParser ${CMAKE_CURRENT_SOURCE_DIR}/LibertyParse.y ${CMAKE_CURRENT_SOURCE_DIR}/LibertyParse.cc)
# add_flex_bison_dependency(LibertyLex LibertyParser)


# # LibertyExpr scan/parse.
# flex_target(LibertyExprLex ${CMAKE_CURRENT_SOURCE_DIR}/LibertyExprLex.l ${CMAKE_CURRENT_SOURCE_DIR}/LibertyExprLex.cc)
# bison_target(LibertyExprParser ${CMAKE_CURRENT_SOURCE_DIR}/LibertyExprParse.y ${CMAKE_CURRENT_SOURCE_DIR}/LibertyExprParse.cc)
# add_flex_bison_dependency(LibertyExprLex LibertyExprParser)

aux_source_directory(./ SRC)

add_library(liberty ${SRC} 
    ${HOME_THIRDPARTY}/parser/liberty/mLibertyEquivCells.cc 
    ${HOME_THIRDPARTY}/parser/liberty/mLibertyExprLex.cc 
    ${HOME_THIRDPARTY}/parser/liberty/mLibertyExprParse.cc
    ${HOME_THIRDPARTY}/parser/liberty/mLibertyLex.cc
    ${HOME_THIRDPARTY}/parser/liberty/mLibertyParse.cc
    ${HOME_THIRDPARTY}/parser/liberty/mLibertyExpr.cc
)

# add_library(liberty ${SRC} ${FLEX_LibertyLex_OUTPUTS} ${BISON_LibertyParser_OUTPUTS} 
# ${FLEX_LibertyExprLex_OUTPUTS} ${BISON_LibertyExprParser_OUTPUTS})

include(ExternalProject)

set(RUST_PROJECT_NAME liberty_parser)
set(RUST_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib-rust/liberty-parser)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/debug/lib${RUST_PROJECT_NAME}.so)
    set(RUST_BUILD_CMD_OPTION "")
else()
    set(RUST_LIB_PATH ${RUST_PROJECT_DIR}/target/release/lib${RUST_PROJECT_NAME}.so)
    set(RUST_BUILD_CMD_OPTION "--release")
endif()

message("liberty parser rust lib path ${RUST_LIB_PATH}")

include(ExternalProject)

ExternalProject_Add(
    ${RUST_PROJECT_NAME}
    PREFIX ${RUST_PROJECT_DIR}
    SOURCE_DIR ${RUST_PROJECT_DIR}
    BINARY_DIR ${RUST_PROJECT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cargo build ${RUST_BUILD_CMD_OPTION}
    INSTALL_COMMAND ""
    BUILD_ALWAYS 1
    BUILD_BYPRODUCTS ${RUST_LIB_PATH}
)

add_dependencies(liberty ${RUST_PROJECT_NAME})

target_link_libraries(liberty str sta-solver log ${RUST_LIB_PATH})

target_include_directories(liberty PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${HOME_THIRDPARTY}/parser/liberty/)
