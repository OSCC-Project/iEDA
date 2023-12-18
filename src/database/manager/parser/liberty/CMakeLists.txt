cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)


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

# ADD_LIBRARY(liberty ${SRC} ${FLEX_LibertyLex_OUTPUTS} ${BISON_LibertyParser_OUTPUTS} 
# ${FLEX_LibertyExprLex_OUTPUTS} ${BISON_LibertyExprParser_OUTPUTS})

target_link_libraries(liberty str sta-solver log)
target_include_directories(liberty PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${HOME_THIRDPARTY}/parser/liberty/)