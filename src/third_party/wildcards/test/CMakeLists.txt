project(Test)

add_executable(selftest
  src/cx/array_test.cpp
  src/cx/string_view_test.cpp
  src/cx/tuple_test.cpp
  src/cx/utility_test.cpp
  src/wildcards/match_test.cpp
  src/wildcards/matcher_test.cpp
  src/catch.cpp
)

target_include_directories(selftest PRIVATE ../include include)

if(WILDCARDS_CXX_STANDARD EQUAL 17)
  message(STATUS "Enabling C++17 for selftest")
elseif(WILDCARDS_CXX_STANDARD EQUAL 14)
  message(STATUS "Enabling C++14 for selftest")
endif()

set_target_properties(selftest
  PROPERTIES
    CXX_STANDARD ${WILDCARDS_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^GNU")
  target_compile_options(selftest
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wcast-align
      -Wconversion
      -Wdouble-promotion
      -Wformat=2
      -Wlogical-op
      -Wnon-virtual-dtor
      -Wold-style-cast
      -Woverloaded-virtual
      -Wshadow
      -Wsign-conversion
      -Wsign-promo
      -Wunused
      -Wuseless-cast
  )

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6.0)
    target_compile_options(selftest
      PRIVATE
        -Wduplicated-cond
        -Wmisleading-indentation
        -Wnull-dereference
    )
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
    target_compile_options(selftest PRIVATE -Wduplicated-branches)
  endif()

  if(WILDCARDS_WARNINGS_AS_ERRORS)
    target_compile_options(selftest PRIVATE -Werror)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^AppleClang|^Clang")
  target_compile_options(selftest
    PRIVATE
      -Weverything
      -Wno-c++98-compat
      -Wno-missing-prototypes
      -Wno-reserved-id-macro
      -Wno-padded
  )

  if(WILDCARDS_CXX_STANDARD GREATER_EQUAL 14)
    target_compile_options(selftest PRIVATE -Wno-covered-switch-default)
  endif()

  if(WILDCARDS_WARNINGS_AS_ERRORS)
    target_compile_options(selftest PRIVATE -Werror)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^MSVC")
  target_compile_options(selftest PRIVATE /W4)

  if(WILDCARDS_WARNINGS_AS_ERRORS)
    target_compile_options(selftest PRIVATE /WX)
  endif()
endif()

add_test(NAME SelfTest COMMAND selftest)

if(WILDCARDS_CLANGFORMAT)
  include(ClangFormat)

  clangformat_setup(
    src/cx/array_test.cpp
    src/cx/string_view_test.cpp
    src/cx/tuple_test.cpp
    src/cx/utility_test.cpp
    src/wildcards/match_test.cpp
    src/wildcards/matcher_test.cpp
  )
endif()
