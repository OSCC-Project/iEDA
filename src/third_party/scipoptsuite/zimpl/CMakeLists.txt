cmake_minimum_required(VERSION 3.3)

project(ZIMPL
    VERSION 3.5.3
    LANGUAGES C)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # if changing these flags, also update GCCWARN/GXXWARN in make/make.project
    set(ADD_C_FLAGS -Wall -Wextra -Wno-unknown-pragmas -Wpointer-arith -Wcast-align -Wwrite-strings -Winline -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wmissing-noreturn -Wmissing-declarations -fno-omit-frame-pointer)

    add_compile_options(${ADD_C_FLAGS})
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        set(ADD_GNU_FLAGS -Wsuggest-attribute=format -Wno-suggest-attribute=noreturn )
        add_compile_options(${ADD_GNU_FLAGS})
        if (CMAKE_C_COMPILER_VERSION VERSION_LESS 5)
            message(WARNING "GCC version smaller than 5, it is recommended to use version 10 or higher.")
            add_compile_options(-Wno-nonnull-compare -Wno-unused-parameter -Wno-unused-function -Wno-pragmas -Wno-clobbered -Wno-attributes)
        elseif (CMAKE_C_COMPILER_VERSION VERSION_LESS 8)
            message(WARNING "GCC version smaller than 8, it is recommended to use version 10 or higher.")
            add_compile_options(-Wno-nonnull-compare -Wno-unused-parameter -Wno-unused-function -fstack-protector-strong -Wduplicated-branches)
        elseif (CMAKE_C_COMPILER_VERSION VERSION_LESS 9)
            message(WARNING "GCC version smaller than 9, it is recommended to use version 10 or higher.")
            add_compile_options(-Wno-nonnull-compare -Wno-unused-parameter -Wno-unused-function -fstack-protector-strong -Wduplicated-branches)
        else() # gcc 10 or later
            add_compile_options(-Wno-unused-parameter -Wno-unused-function -Wsuggest-attribute=const -Wsuggest-attribute=pure -Wsuggest-attribute=malloc -Wsuggest-attribute=cold -Wstrict-overflow=4 -fstack-protector-strong -Wduplicated-branches -Wno-nonnull-compare)
        endif()
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(ADD_CLANG_FLAGS -Wno-unused-parameter -Wno-unused-function -Wno-cast-align -Wno-strict-prototypes -Wno-pragmas -Wno-attributes -Wstrict-overflow=4 -fstack-protector-strong -Wno-invalid-command-line-argument)
        set(EXTRA_CLANG_FLAGS -Weverything -Wno-reserved-id-macro -Wno-cast-qual -Wno-bad-function-cast -Wno-float-equal -Wno-unused-macros -Wno-implicit-fallthrough -Wno-switch-enum -Wno-padded -Wno-covered-switch-default -Wno-shorten-64-to-32 -Wno-pedantic -Wno-format-nonliteral -Wno-disabled-macro-expansion -Wno-static-in-inline)
        add_compile_options(${ADD_CLANG_FLAGS} ${EXTRA_CLANG_FLAGS})
        if (CMAKE_C_COMPILER_VERSION VERSION_LESS 7)
            message(WARNING "Clang version smaller than 7, it is recommended to use version 10 or higher.")
        else()
            add_compile_options(-Wno-extra-semi-stmt)
        endif()
    endif()
endif()
if(CMAKE_C_COMPILER MATCHES "intel")
    add_compile_options(-wd111,151,171,279,981,1173,1419,1684,2259)
endif()

# use C99 standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

option(ZLIB "use ZLIB" ON)
option(SANITIZE_ADDRESS "should the address sanitizer be enabled in debug mode if available" OFF)
option(SANITIZE_MEMORY "should the memory sanitizer be enabled in debug mode if available" OFF)
option(SANITIZE_UNDEFINED "should the undefined behavior sanitizer be enabled in debug mode if available" OFF)
option(SANITIZE_THREAD "should the thread sanitizer be enabled in debug mode if available" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-DVERSION=\"${ZIMPL_VERSION_MAJOR}.${ZIMPL_VERSION_MINOR}.${ZIMPL_VERSION_PATCH}\")

# path to e.g. findGMP module
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# On macOS, search Homebrew for keg-only versions of Bison and Flex. Xcode does
# not provide new enough versions for us to use.
if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    find_program(BREW brew)
    if(DEFINED BREW)
        execute_process(
            COMMAND ${BREW} --prefix bison
            RESULT_VARIABLE BREW_BISON
            OUTPUT_VARIABLE BREW_BISON_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (BREW_BISON EQUAL 0 AND EXISTS "${BREW_BISON_PREFIX}")
            message(STATUS "Found Bison keg installed by Homebrew at ${BREW_BISON_PREFIX}")
            set(BISON_EXECUTABLE "${BREW_BISON_PREFIX}/bin/bison")
        endif()

        execute_process(
            COMMAND ${BREW} --prefix flex
            RESULT_VARIABLE BREW_FLEX
            OUTPUT_VARIABLE BREW_FLEX_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (BREW_FLEX EQUAL 0 AND EXISTS "${BREW_FLEX_PREFIX}")
            message(STATUS "Found Flex keg installed by Homebrew at ${BREW_FLEX_PREFIX}")
            set(FLEX_EXECUTABLE "${BREW_FLEX_PREFIX}/bin/flex")
        endif()
    endif()
endif()

find_package(BISON 2.4.0 REQUIRED)
find_package(FLEX REQUIRED)
find_package(GMP REQUIRED)

# make 'Release' the default build type
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# filter /MD and /MDd from C and CXX flags
if(MSVC)
    set(variables
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )

    foreach(variable ${variables})
       string(REGEX REPLACE "/M[T,D][ d]" "" ${variable} "${${variable}}")
       # message("${variable} = ${${variable}}")
    endforeach()
endif()

include_directories(${GMP_INCLUDE_DIRS})
set(libs ${libs} ${GMP_LIBRARIES})

if(ZLIB)
    find_package(ZLIB)
endif()
if(ZLIB_FOUND)
    set(libs ${libs} ${ZLIB_LIBRARIES})
    include_directories(${ZLIB_INCLUDE_DIRS})
else()
    add_definitions(-DWITHOUT_ZLIB)

    # look for pcre if ZLIB could not be found
    find_package(PCRE)
    if(PCRE_FOUND)
        add_definitions(-DWITH_PCRE)
        add_definitions(-DPCRE2_STATIC)
        set(libs ${libs} ${PCRE_LIBRARIES})
        include_directories(${PCRE_INCLUDE_DIRS})
    endif()
endif()

if(MSVC)
   add_definitions(-Dpopen=_popen)
   add_definitions(-Dpclose=_pclose)
   add_definitions(-Dinline=_inline)
endif()

include(CheckSymbolExists)

# link to math library if it is available
find_library(libm m)
if(NOT libm)
  set(libm "")
endif()

set(libs ${libs} ${libm})

add_subdirectory(src)
