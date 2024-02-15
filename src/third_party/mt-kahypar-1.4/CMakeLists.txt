cmake_minimum_required(VERSION 3.16)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(MtKaHyPar CXX C)
set(PROJECT_VENDOR "Tobias Heuer")
set(PROJECT_CONTACT "tobias.heuer@kit.edu")
set(PROJECT_URL "https://github.com/kittobi1992/mt-kahypar")
set(PROJECT_DESCRIPTION "Mt-KaHyPar: Multi-Threaded Karlsruhe Hypergraph Partitioning")
set(PROJECT_VERSION "1.0.0")
set(CMAKE_CXX_STANDARD 17)
# CMake Options

option(KAHYPAR_DOWNLOAD_BOOST
  "Download boost automatically and compile required libraries." OFF)

option(KAHYPAR_DOWNLOAD_TBB
  "Download TBB automatically." ON)

option(KAHYPAR_ENFORCE_MINIMUM_TBB_VERSION
  "Enforces the minimum required TBB version." ON)

option(KAHYPAR_USE_GCOV
  "Compile and run tests with gcov for coverage analysis." OFF)

option(KAHYPAR_DISABLE_ASSERTIONS
  "Disable KaHyPar's internal assertions." OFF)

option(KAHYPAR_USE_STANDARD_ASSERTIONS
  "Use standard C++ asserts instead of custom assertions." OFF)

option(KAHYPAR_ENABLE_HEAVY_PREPROCESSING_ASSERTIONS
  "Enable heavy assertions in preprocessing phase." OFF)

option(KAHYPAR_ENABLE_HEAVY_COARSENING_ASSERTIONS
  "Enable heavy assertions in coarsening phase." OFF)

option(KAHYPAR_ENABLE_HEAVY_INITIAL_PARTITIONING_ASSERTIONS
  "Enable heavy assertions in initial partitioning phase." OFF)

option(KAHYPAR_ENABLE_HEAVY_REFINEMENT_ASSERTIONS
  "Enable heavy assertions in refinement phase." OFF)

option(KAHYPAR_USE_64_BIT_IDS
  "Enables 64-bit vertex and hyperedge IDs." OFF)

option(KAHYPAR_TRAVIS_BUILD
  "Indicate that this build is executed on Travis CI." OFF)

option(KAHYPAR_CI_BUILD
  "Indicate that this build is executed on GitHub Actions." OFF)

option(KAHYPAR_ADD_ADDRESS_SANITIZER
  "Adds address sanitizer to compile options." OFF)

option(KAHYPAR_ENABLE_THREAD_PINNING
  "Enables thread pinning in Mt-KaHyPar." ON)

option(KAHYPAR_ENABLE_ARCH_COMPILE_OPTIMIZATIONS
  "Adds the compile flags `-mtune=native -march=native`" ON)

option(KAHYPAR_ENABLE_EXPERIMENTAL_FEATURES
  "Enables some experimental features. Can be turned off for faster compilation." OFF)

option(KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES
  "Enables graph partitioning features. Can be turned off for faster compilation." ON)

option(KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
  "Enables highest quality preset features. Can be turned off for faster compilation." ON)

option(KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES
  "Enables large k partitioning features. Can be turned off for faster compilation." ON)

option(KAHYPAR_ENABLE_SOED_METRIC
  "Enables the sum-of-external-degree metric. Can be turned off for faster compilation." ON)

option(KAHYPAR_ENABLE_STEINER_TREE_METRIC
  "Enables the Steiner tree metric. Can be turned off for faster compilation." ON)

if(KAHYPAR_DISABLE_ASSERTIONS)
  add_compile_definitions(KAHYPAR_DISABLE_ASSERTIONS)
endif(KAHYPAR_DISABLE_ASSERTIONS)

if(KAHYPAR_USE_STANDARD_ASSERTIONS)
  add_compile_definitions(KAHYPAR_USE_STANDARD_ASSERTIONS)
endif(KAHYPAR_USE_STANDARD_ASSERTIONS)

if(KAHYPAR_ENABLE_HEAVY_PREPROCESSING_ASSERTIONS)
  add_compile_definitions(KAHYPAR_ENABLE_HEAVY_PREPROCESSING_ASSERTIONS)
endif(KAHYPAR_ENABLE_HEAVY_PREPROCESSING_ASSERTIONS)

if(KAHYPAR_ENABLE_HEAVY_COARSENING_ASSERTIONS)
  add_compile_definitions(KAHYPAR_ENABLE_HEAVY_COARSENING_ASSERTIONS)
endif(KAHYPAR_ENABLE_HEAVY_COARSENING_ASSERTIONS)

if(KAHYPAR_ENABLE_HEAVY_INITIAL_PARTITIONING_ASSERTIONS)
  add_compile_definitions(KAHYPAR_ENABLE_HEAVY_INITIAL_PARTITIONING_ASSERTIONS)
endif(KAHYPAR_ENABLE_HEAVY_INITIAL_PARTITIONING_ASSERTIONS)

if(KAHYPAR_ENABLE_HEAVY_REFINEMENT_ASSERTIONS)
  add_compile_definitions(KAHYPAR_ENABLE_HEAVY_REFINEMENT_ASSERTIONS)
endif(KAHYPAR_ENABLE_HEAVY_REFINEMENT_ASSERTIONS)

if(KAHYPAR_USE_64_BIT_IDS)
  add_compile_definitions(KAHYPAR_USE_64_BIT_IDS)
endif(KAHYPAR_USE_64_BIT_IDS)

if(KAHYPAR_TRAVIS_BUILD)
  add_compile_definitions(KAHYPAR_TRAVIS_BUILD)
endif(KAHYPAR_TRAVIS_BUILD)

if(KAHYPAR_ENABLE_THREAD_PINNING)
  add_compile_definitions(KAHYPAR_ENABLE_THREAD_PINNING)
endif(KAHYPAR_ENABLE_THREAD_PINNING)

if(KAHYPAR_ENABLE_EXPERIMENTAL_FEATURES)
  add_compile_definitions(KAHYPAR_ENABLE_EXPERIMENTAL_FEATURES)
endif(KAHYPAR_ENABLE_EXPERIMENTAL_FEATURES)

if(KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES)
  add_compile_definitions(KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES)
endif(KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES)

if(KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES)
  add_compile_definitions(KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES)
endif(KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES)

if(KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES)
  add_compile_definitions(KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES)
endif(KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES)

if(KAHYPAR_ENABLE_SOED_METRIC)
  add_compile_definitions(KAHYPAR_ENABLE_SOED_METRIC)
endif(KAHYPAR_ENABLE_SOED_METRIC)

if(KAHYPAR_ENABLE_STEINER_TREE_METRIC)
  add_compile_definitions(KAHYPAR_ENABLE_STEINER_TREE_METRIC)
endif(KAHYPAR_ENABLE_STEINER_TREE_METRIC)

include_directories(${PROJECT_SOURCE_DIR})
find_package(Threads REQUIRED)
message(STATUS "Found Threads: ${CMAKE_THREAD_LIBS_INIT}")

# Include Submodules
add_subdirectory(external_tools/googletest EXCLUDE_FROM_ALL)
include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(SYSTEM ${gtest_SOURCE_DIR}/../googlemock/include ${gtest_SOURCE_DIR}/../googlemock/)
include_directories(external_tools/kahypar-shared-resources)
include_directories(external_tools/growt)
include_directories(external_tools/WHFC)
include_directories(external_tools/pcg)

if(KAHYPAR_DOWNLOAD_BOOST)
  # Download Boost
  execute_process(COMMAND cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_boost.cmake)
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/external_tools/boost/)
  file(GLOB MINI_BOOST_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/external_tools/boost/libs/program_options/src/*.cpp)

  add_library(mini_boost STATIC ${MINI_BOOST_SOURCES})
  set_target_properties(mini_boost PROPERTIES LINKER_LANGUAGE CXX)
  set(Boost_LIBRARIES mini_boost)
  set(Boost_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/external_tools/boost/boost/)
  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
else()
  # Find Boost Program Options Library
  find_package(Boost 1.69 REQUIRED COMPONENTS program_options)
  if(Boost_FOUND)
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    set(KAHYPAR_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${KAHYPAR_INCLUDE_DIRS})
    set(KAHYPAR_LINK_LIBRARIES ${Boost_LIBRARIES} ${KAHYPAR_LINK_LIBRARIES})
    message(STATUS "Boost Include: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost Library Dirs: ${Boost_LIBRARY_DIRS}")
    message(STATUS "Boost Libraries: ${Boost_LIBRARIES}")
    if(WIN32)
      add_definitions(-DBOOST_ALL_NO_LIB)
      add_definitions(-DBOOST_PROGRAM_OPTIONS_DYN_LINK=1)
    endif()
  else()
    MESSAGE(FATAL_ERROR "
      Boost not found. Install Boost on your system or
      add -DKAHYPAR_DOWNLOAD_BOOST=On to the cmake build command.")
  endif()
endif()

# Download TBB
# if(KAHYPAR_DOWNLOAD_TBB)
#   if ( UNIX AND NOT WIN32 )
#     execute_process(COMMAND cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_tbb_linux.cmake)
#     set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/external_tools/tbb)
#   elseif ( MSVC )
#     execute_process(COMMAND cmake -G ${CMAKE_GENERATOR} -P ${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_tbb_windows.cmake)
#     set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/external_tools/tbb)
#   else()
#     MESSAGE(WARNING "
#       We did not find a TBB version for your platform online.
#       The build uses the TBB version installed on your system.")
#   endif()
# endif()

set(TBB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external_tools/tbb)
# Find TBB Library
find_package(TBB REQUIRED)
if (TBB_FOUND)
  if ((TBB_INTERFACE_VERSION GREATER_EQUAL 12050) OR (NOT KAHYPAR_ENFORCE_MINIMUM_TBB_VERSION) OR KAHYPAR_CI_BUILD)
    include_directories(SYSTEM ${TBB_INCLUDE_DIRS} ${TBB_MALLOC_INCLUDE_DIRS})
    link_libraries(${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARIES})
    MESSAGE(STATUS "TBB Version: ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} (Interface Version=${TBB_INTERFACE_VERSION})")
    MESSAGE(STATUS "Found TBB library: inc=${TBB_INCLUDE_DIRS}, lib=${TBB_LIBRARIES}")
    MESSAGE(STATUS "Found TBB Malloc Proxy library: inc=${TBB_MALLOC_PROXY_INCLUDE_DIRS}, lib=${TBB_MALLOC_PROXY_LIBRARIES}")
  ELSE ()
    MESSAGE(FATAL_ERROR "
      Your current TBB version is too old. Your interface version is ${TBB_INTERFACE_VERSION}, but the minimum
      required version is 12050 (OneTBB 2021.5). You can download a suitable version by adding
      -DKAHYPAR_DOWNLOAD_TBB=ON to the cmake build command (may require to clear the cmake cache).
      Note that Mt-KaHyPar also compiles with older versions of TBB. However, we found some unexpected
      behavior of a TBB function that causes on our side segmentation faults in really rare cases.
      If you want to ignore these warnings, you can add -DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF to the
      cmake build command which will build Mt-KaHyPar with your current TBB version.")
  ENDIF()
ELSE ()
  MESSAGE(FATAL_ERROR "
    TBB library not found. Install TBB on your system or
    add -DKAHYPAR_DOWNLOAD_TBB=On to the cmake build command
    (only available for Linux and MSVC).")
ENDIF()

# Find HWLOC Library
FIND_PATH(HWLOCK_INCLUDE_DIR NAME hwloc.h
  HINTS $ENV{HOME}/local/include /opt/local/include /usr/local/include /usr/include)
FIND_LIBRARY(HWLOCK_LIBRARY NAME hwloc
  HINTS $ENV{HOME}/local/lib64 $ENV{HOME}/local/lib /usr/local/lib64 /usr/local/lib /opt/local/lib64 /opt/local/lib /usr/lib64 /usr/lib
)

IF (HWLOCK_INCLUDE_DIR AND HWLOCK_LIBRARY)
  MESSAGE(STATUS "Found hwlock library: inc=${HWLOCK_INCLUDE_DIR}, lib=${HWLOCK_LIBRARY}")
  include_directories(${HWLOCK_INCLUDE_DIR})
  link_libraries(${HWLOCK_LIBRARY})
ELSE ()
  MESSAGE(FATAL_ERROR "
    HwLoc library not found. Install HwLoc on your system.")
ENDIF ()

# Add targets for code coverage analysis
if(KAHYPAR_USE_GCOV)

  include(CodeCoverage)
  setup_target_for_coverage(${PROJECT_NAME}_coverage tests coverage)

  # find programs
  find_program(GENHTML genhtml)
  find_program(LCOV lcov)

  if(NOT LCOV OR NOT GENHTML)
    message(SEND_ERROR "Coverage analysis requires lcov and genhtml.")
  endif()

  # add coverage anaylsis compile and link flags
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")

  # add cached variable containing parameters for lcov/genhtml
  set(LCOV_FLAGS "" CACHE STRING "parameters for lcov")
  set(GENHTML_FLAGS --legend --no-branch-coverage
    CACHE STRING "parameters for genhtml")
  set(COVERAGE_LCOV_EXCLUDES "external_tools/*" )

  # custom target to run before tests
  add_custom_target(lcov-reset
    COMMAND ${LCOV} -q --directory ${CMAKE_BINARY_DIR} --zerocounters
    COMMENT "Resetting code coverage counters")

  # custom lcov target to run tests
  add_custom_target(lcov-runtests
    COMMAND make
    DEPENDS lcov-reset
    COMMENT "Running all unit tests")

  # get git version description
  execute_process(COMMAND git describe --tags
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GITDESC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # command sequence to gather, clean and generate HTML coverage report
  add_custom_target(lcov-html
    COMMAND ${LCOV} -q --directory . --capture --output-file lcov.info
    COMMAND ${LCOV} -q --remove lcov.info '/usr/*' '*/extlib/*' ${LCOV_FLAGS} --output-file lcov-clean.info
    COMMAND ${GENHTML} -q -o coverage --title "KaHyPar ${GITDESC}" --prefix ${PROJECT_SOURCE_DIR} ${GENHTML_FLAGS} lcov-clean.info
    DEPENDS lcov-runtests
    COMMENT "Capturing code coverage counters and create HTML coverage report"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # top-level target to run tests and generate coverage report
  add_custom_target(test-coverage
    COMMENT "Generate HTML coverage report"
    DEPENDS lcov-html)

endif(KAHYPAR_USE_GCOV)

# Set default build type if not set
if(NOT CMAKE_BUILD_TYPE)
  set( CMAKE_BUILD_TYPE Release CACHE STRING
       "Choose the type of build, options are: Debug, Release, RelWithDebInfo"
       FORCE )
endif()

# Get current commit hash
find_package(Git)
include(GetGitRevisionDescription)
get_git_head_revision(MT_KAHYPAR_VERSION_GIT_REFSPEC MT_KAHYPAR_VERSION_GIT_SHA1)
if(MT_KAHYPAR_VERSION_GIT_REFSPEC)
  message(STATUS "Detected git refspec ${MT_KAHYPAR_VERSION_GIT_REFSPEC} sha ${MT_KAHYPAR_VERSION_GIT_SHA1}")
  configure_file(${PROJECT_SOURCE_DIR}/mt-kahypar/application/git_revision.txt.in ${PROJECT_BINARY_DIR}/mt-kahypar/application/git_head.txt)
endif(MT_KAHYPAR_VERSION_GIT_REFSPEC)

if(NOT MSVC)
  # Add compile flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPARANOID ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcx16")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

  # lld is way faster than ld. If you have it, use it!
  find_program(LLD_BIN lld)
  if (LLD_BIN)
    message(STATUS "Found and will use LLVM linker " ${LLD_BIN})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=lld")
  else()
    message(STATUS "Default linker")
  endif()

  include(CheckSSE4_2)
  if( BUILTIN_POPCNT )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
  endif()

  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-mcrc32 KAHYPAR_HAS_CRC32)
  if(KAHYPAR_HAS_CRC32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcrc32")
  endif()


  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 ")

  if(UNIX AND NOT WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -fsanitize=undefined")
  endif()

  if(KAHYPAR_ENABLE_ARCH_COMPILE_OPTIMIZATIONS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native -march=native")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -mtune=native -march=native")
  endif()

  if(KAHYPAR_ADD_ADDRESS_SANITIZER AND (NOT KAHYPAR_CI_BUILD))
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")
  endif()

  if(ENABLE_PROFILE MATCHES ON)
    message(STATUS "Profiling activated")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DENABLE_PROFILE")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g3 -DENABLE_PROFILE -fno-omit-frame-pointer")
    set(PROFILE_FLAGS "-lprofiler")
  endif()
elseif(MSVC)
  message(FATAL_ERROR "MSVC not supported at the moment")
endif()

add_custom_target(AnalyzeModifiedSources  perl "${PROJECT_SOURCE_DIR}/codestyle/analyze-source.pl" "-w")
add_custom_target(AnalyzeAllSources  perl "${PROJECT_SOURCE_DIR}/codestyle/analyze-source.pl" "-aw")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

include(gmock)
enable_testing()

# add_subdirectory(mt-kahypar/application)
add_subdirectory(tools)
add_subdirectory(lib)
# add_subdirectory(tests)
# add_subdirectory(python)

# This adds the source files. It's important that this happens after the compile targets have been added
add_subdirectory(mt-kahypar)
