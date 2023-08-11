cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELEASE}")

project(GCG)

set(GCG_VERSION_MAJOR 3)
set(GCG_VERSION_MINOR 5)
set(GCG_VERSION_PATCH 3)
set(GCG_VERSION_SUB 0)
set(GCG_VERSION_API 2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# path to e.g. FindGMP module
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(SHARED "Build shared libraries" ON)
set(BUILD_SHARED_LIBS ${SHARED})
message(STATUS "Build shared libraries: " ${SHARED})

# make 'Release' the default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(OPENMP "should gcg be compiled with openmp" OFF)
option(STATIC_GMP "prefer static gmp library" OFF)
option(GSL "should gcg be compiled with GSL" ON)
option(GMP "should gmp be linked" ON)
option(CLIQUER "should cliquer be linked" ON)
option(CPLEX "should Cplex be used for solving pricing MIPs" OFF)
option(HMETIS "should hmetis be called" ON)
option(MT "use static runtime libraries for Visual Studio compiler" OFF)
option(GCG_DEV_BUILD "compile SCIP and SoPlex located in lib/scip-git and lib/soplex-git" OFF)

# OPENMP
if(OPENMP)
   find_package(OpenMP)
   if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   endif()
endif()

# GSL
if(GSL)
   find_package(GSL 2.0)
   if(GSL_FOUND)
      include_directories(${GSL_INCLUDE_DIRS})
      add_definitions(-DWITH_GSL)
      set(GSL_PIC_LIBRARIES ${GSL_LIBRARIES})
   else()
      set(GSL_LIBRARIES)
      set(GSL_PIC_LIBRARIES)
   endif()
endif()

# GMP
if(GMP)
   find_package(GMP)
endif()
if(GMP_FOUND)
   include_directories(${GMP_INCLUDE_DIRS})
   add_definitions(-DWITH_GMP)
else()
   set(GMP_LIBRARIES)
endif()

# CLIQUER
if(CLIQUER)
   find_package(CLIQUER)
endif()
if(CLIQUER_FOUND)
   include_directories(${CLIQUER_INCLUDE_DIRS})
   add_definitions(-DWITH_CLIQUER)
   set(CLIQUER_PIC_LIBRARIES ${CLIQUER_LIBRARIES})
else()
   set(CLIQUER_LIBRARIES)
   set(CLIQUER_PIC_LIBRARIES)
endif()

# CPLEX
if(CPLEX)
   find_package(CPLEX REQUIRED)
endif()
if(CPLEX_FOUND)
   include_directories(${CPLEX_INCLUDE_DIRS})
   set(CPLEX_PIC_LIBRARIES ${CPLEX_LIBRARIES})
else()
   set(CPLEX_LIBRARIES)
   set(CPLEX_PIC_LIBRARIES)
endif()

# HMETIS
if(HMETIS)
   find_package(HMETIS)
endif()
if(HMETIS_FOUND)
   add_definitions(-DWITH_HMETIS)
endif()

set(SYM bliss CACHE STRING "options for symmetry computation")  #create the variable
set_property(CACHE SYM PROPERTY STRINGS bliss none )  #define list of values GUI will offer for the variable

#set the correct rpath for OS X
set(CMAKE_MACOSX_RPATH ON)

#set defines for Windows
if(WIN32)
    add_definitions(-DNO_SIGACTION)
    add_definitions(-DNO_STRTOK_R)
endif()
if(MSVC)
    add_definitions(/wd4100)
    add_definitions(/wd4244)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Visual Studio compiler with static runtime libraries
if(MSVC AND MT)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif()

# create a target for updating the current git hash
file(WRITE ${CMAKE_BINARY_DIR}/gcg_update_githash.cmake "
find_program(GIT git)
if(EXISTS \${DST})
   file(STRINGS \${DST} GITHASH_OLD)
   string(REGEX REPLACE \"#define GCG_GITHASH \\\"(.*)\\\"\" \"\\\\1\" GITHASH_OLD \${GITHASH_OLD})
endif()
if((GIT) AND (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git))
   execute_process(
      COMMAND \${GIT} describe --always --dirty
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE GITHASH OUTPUT_STRIP_TRAILING_WHITESPACE)
   string(REGEX REPLACE \"^.*-g\" \"\" GITHASH \${GITHASH})
   if(NOT \${GITHASH} STREQUAL \"\${GITHASH_OLD}\")
      file(WRITE \${DST} \"#define GCG_GITHASH \\\"\${GITHASH}\\\"\n\")
   endif()
else()
   set(GITHASH \${GITHASH_OLD})
endif()
message(STATUS \"Git hash: \" \${GITHASH})
")
add_custom_target(gcg_update_githash
                  COMMAND ${CMAKE_COMMAND} -DDST=${PROJECT_SOURCE_DIR}/src/githash.c
                                           -P ${CMAKE_BINARY_DIR}/gcg_update_githash.cmake)
# use C++11 standard
set(CMAKE_CXX_STANDARD 11)

# SCIP
if(NOT GCG_DEV_BUILD)
   find_package(SCIP CONFIG REQUIRED)
   if(SCIP_FOUND)
      # SCIP headers can be directly included
      include_directories(${SCIP_INCLUDE_DIRS})
   endif()
endif()

if(NOT DEFINED GCG_DEV_BUILD_SOPLEX_PATH)
   set(GCG_DEV_BUILD_SOPLEX_PATH ${PROJECT_SOURCE_DIR}/lib/soplex-git)
endif()

if(NOT DEFINED GCG_DEV_BUILD_SCIP_PATH)
   set(GCG_DEV_BUILD_SCIP_PATH ${PROJECT_SOURCE_DIR}/lib/scip-git)
endif()

if((NOT SCIP_FOUND) AND (EXISTS ${GCG_DEV_BUILD_SCIP_PATH}) AND (EXISTS ${GCG_DEV_BUILD_SOPLEX_PATH}))
   message(STATUS "Using '${GCG_DEV_BUILD_SOPLEX_PATH}'")
   message(STATUS "Using '${GCG_DEV_BUILD_SCIP_PATH}'")
   set(SOPLEX_DIR ${CMAKE_BINARY_DIR})
   add_subdirectory(${GCG_DEV_BUILD_SOPLEX_PATH})
   add_subdirectory(${GCG_DEV_BUILD_SCIP_PATH})
   set(SCIP_LIBRARIES libscip)
elseif(GCG_DEV_BUILD)
   message(WARNING "GCG_DEV_BUILD option is set but could not find SCIP and SoPlex in the lib/ subdirectory. Please specify GCG_DEV_BUILD_SOPLEX_PATH and GCG_DEV_BUILD_SCIP_PATH")
endif()

#search the selected symmetry computation program
if(SYM STREQUAL "bliss")
   set(SYM_LIBRARIES libbliss ${GMP_LIBRARIES})
   set(SYM_PIC_LIBRARIES libbliss ${GMP_LIBRARIES})
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLISS_DEFINITIONS} -DWITH_BLISS")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BLISS_DEFINITIONS} -DWITH_BLISS")
elseif(SYM STREQUAL "none")
   set(SYM_LIBRARIES)
   set(SYM_PIC_LIBRARIES)
else()
   message(FATAL_ERROR "option SYM has wrong value")
endif()

# TODO: In SCIP, license is in COPYING file and not in LICENSE.
# TODO: SCIP has src/scipbuildflag.c which adds a function that returns the used build flags. Do we also want this?
# TODO: Symmetry handling is separate from the remaining code: there is a symmetry/* code that is compiled differently depending on whether BLISS was found or not. the other tools (presol_sym*) is always included, but can query the symmetry code whether it can actually detect symmetry. This is slightly cleaner from a build system point of view.

set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${SCIP_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${SCIP_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${SCIP_VERSION_PATCH}")
set(CPACK_PACKAGE_VENDOR "Operations Research, RWTH Aachen University and Zuse Institute Berlin (ZIB)")
set(CPACK_PACKAGE_CONTACT "http://www.or.rwth-aachen.de/gcg")
include(CPack)

# go to src/ and compile the code
add_subdirectory(src)

#
# we set the GCG_DIR variable explicitly for the following examples/applications and unittests that depend on GCG.
#
set(GCG_DIR ${CMAKE_BINARY_DIR})
#
# add GCG tests
#
if(NOT TARGET check)
   add_custom_target(check)
endif()
add_subdirectory(check)

#
# add GCG documentation
#
if(NOT TARGET doc)
   add_custom_target(doc)
endif()
add_subdirectory(doc)

#
# add unit tests as a single target. Including tests will add the unit tests as single executables
#
add_custom_target(gcg_unittests)
#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_custom_target(gcg_all_executables DEPENDS gcg gcg_unittests)

enable_testing()

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
   include(FeatureSummary)
   feature_summary(WHAT ALL)
endif()
