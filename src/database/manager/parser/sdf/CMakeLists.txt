cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)

################################################################
#
# Flex/bison scanner/parsers
#
################################################################

# find_package(FLEX)
# find_package(BISON)

# Sdf scan/parse. 
# flex_target(SdfLex ${CMAKE_CURRENT_SOURCE_DIR}/SdfLex.l ${CMAKE_CURRENT_SOURCE_DIR}/SdfLex.cc COMPILE_FLAGS --prefix=Sdf_) 
# bison_target(SdfParser ${CMAKE_CURRENT_SOURCE_DIR}/SdfParse.y ${CMAKE_CURRENT_SOURCE_DIR}/SdfParse.cc COMPILE_FLAGS --name-prefix=Sdf_) 
# add_flex_bison_dependency(SdfLex SdfParser)

aux_source_directory(./ SRC)

#ADD_LIBRARY(sdf-parser ${SRC} ${FLEX_SdfLex_OUTPUTS} ${BISON_SdfParser_OUTPUTS})
add_library(sdf-parser ${SRC})

target_link_libraries(sdf-parser str log) 