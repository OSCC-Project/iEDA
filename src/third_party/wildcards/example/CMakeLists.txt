project(Example)

function(add_example target file_name)
  add_executable(${target} ${file_name})
  target_include_directories(${target} PRIVATE ../include)

  if(WILDCARDS_CXX_STANDARD EQUAL 17)
    message(STATUS "Enabling C++17 for ${target}")
  elseif(WILDCARDS_CXX_STANDARD EQUAL 14)
    message(STATUS "Enabling C++14 for ${target}")
  endif()

  set_target_properties(${target}
    PROPERTIES
      CXX_STANDARD ${WILDCARDS_CXX_STANDARD}
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "^GNU")
    target_compile_options(${target}
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
      target_compile_options(${target}
        PRIVATE
          -Wduplicated-cond
          -Wmisleading-indentation
          -Wnull-dereference
      )
    endif()

    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
      target_compile_options(${target} PRIVATE -Wduplicated-branches)
    endif()

    if(WILDCARDS_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^AppleClang|^Clang")
    target_compile_options(${target}
      PRIVATE
        -Weverything
        -Wno-c++98-compat
        -Wno-missing-prototypes
        -Wno-reserved-id-macro
        -Wno-padded
    )

    if(WILDCARDS_CXX_STANDARD GREATER_EQUAL 14)
      target_compile_options(${target} PRIVATE -Wno-covered-switch-default)
    endif()

    if(WILDCARDS_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^MSVC")
    target_compile_options(${target} PRIVATE /W4)

    if(WILDCARDS_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()
endfunction()

add_example(example01a example01a.cpp)
add_example(example01b example01b.cpp)
add_example(example01c example01c.cpp)
add_example(example02 example02.cpp)
add_example(example03 example03.cpp)
add_example(example04 example04.cpp)
if(WILDCARDS_CXX_STANDARD EQUAL 17)
  if(CMAKE_CXX_COMPILER_ID MATCHES "^GNU")
    add_example(example05 example05.cpp)
    target_link_libraries(example05 stdc++fs)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.0)
      add_example(example05 example05.cpp)
      target_link_libraries(example05 c++experimental)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^MSVC")
    add_example(example05 example05.cpp)
  endif()
endif()
add_example(example06 example06.cpp)

if(WILDCARDS_CLANGFORMAT)
  include(ClangFormat)

  clangformat_setup(
    example01a.cpp
    example01b.cpp
    example01c.cpp
    example02.cpp
    example03.cpp
    example04.cpp
    example05.cpp
    example06.cpp
  )
endif()
