set(ITF_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${ITF_GENERATED_DIR})

find_package(FLEX)
flex_target(itfScanner
  ${CMAKE_CURRENT_SOURCE_DIR}/itf_lex.l
  ${ITF_GENERATED_DIR}/itf_lex.cpp
  DEFINES_FILE  ${ITF_GENERATED_DIR}/itf_lex.hpp
  # COMPILE_FLAGS "-d" # debug
)
file(COPY itf_lex.l DESTINATION ${CMAKE_BINARY_DIR}) # for debug

find_package(BISON)
bison_target(itfParser
${CMAKE_CURRENT_SOURCE_DIR}/itf_parser.yy
${ITF_GENERATED_DIR}/itf_parser.cpp
DEFINES_FILE  ${ITF_GENERATED_DIR}/itf_parser.hpp
# COMPILE_FLAGS "-t" # debug
)
file(COPY itf_parser.yy DESTINATION ${CMAKE_BINARY_DIR}) # for debug

##-------------------------- Library --------------------------------##
set(SRC
  itfUtil.cpp
  itfiConductor.cpp
  itfiDielectric.cpp
  itfiVia.cpp
  itfrCallBacks.cpp
  itfrData.cpp
  itfrReader.cpp
  itfrSettings.cpp
)

add_library(itf
  STATIC
    ${SRC}
    ${FLEX_itfScanner_OUTPUTS}
    ${BISON_itfParser_OUTPUTS}
)

target_compile_options(itf PRIVATE -Wall -Wextra -pedantic)

target_include_directories(itf
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    ${ITF_GENERATED_DIR}
)
