cmake_minimum_required(VERSION 3.9)

if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
endif()

project(Wildcards VERSION 1.5.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

option(WILDCARDS_BUILD_TESTS "Build unit tests" ON)
option(WILDCARDS_BUILD_EXAMPLES "Build examples" OFF)
option(WILDCARDS_WARNINGS_AS_ERRORS "Enable all warnings as errors" ON)

set(WILDCARDS_CXX_STANDARD 11 CACHE STRING "Build using C++ standard")

option(WILDCARDS_CLANGFORMAT "Setup clangformat target" OFF)

configure_file(
  ${PROJECT_SOURCE_DIR}/include/wildcards.hpp.in
  ${PROJECT_SOURCE_DIR}/include/wildcards.hpp
)

if(NOT_SUBPROJECT)
  if(WILDCARDS_BUILD_TESTS)
    include(CTest)

    add_subdirectory(test)
  endif()

  if(WILDCARDS_BUILD_EXAMPLES)
    add_subdirectory(example)
  endif()
endif()

if(WILDCARDS_CLANGFORMAT)
  include(ClangFormat)

  clangformat_setup(
    include/config.hpp
    include/cx.hpp
    include/wildcards.hpp
    include/cx/algorithm.hpp
    include/cx/array.hpp
    include/cx/functional.hpp
    include/cx/iterator.hpp
    include/cx/string_view.hpp
    include/cx/tuple.hpp
    include/cx/utility.hpp
    include/wildcards/cards.hpp
    include/wildcards/match.hpp
    include/wildcards/matcher.hpp
    include/wildcards/utility.hpp
  )
endif()
