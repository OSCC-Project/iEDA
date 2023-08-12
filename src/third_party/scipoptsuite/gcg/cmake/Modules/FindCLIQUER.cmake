include(FindPackageHandleStandardArgs)

# check whether environment variable CLIQUER_DIR was set
if(NOT CLIQUER_DIR)
   set(CLIQUER_DIR_TEST $ENV{CLIQUER_DIR})
   if(CLIQUER_DIR_TEST)
      set(CLIQUER_DIR $ENV{CLIQUER_DIR} CACHE PATH "Path to cliquer build directory")
   endif()
endif()

# if the cliquer directory is specified, first try to use exactly that cliquer
if(CLIQUER_DIR)
   # look for the includes with subdirectory cliquer
   find_path(CLIQUER_INCLUDE_DIR
       NAMES cliquer/cliquer.h
       PATHS ${CLIQUER_DIR}
       PATH_SUFFIXES include
       NO_DEFAULT_PATH
       )

   # if not found look for the includes without cliquer subdirectory
   if(NOT CLIQUER_INCLUDE_DIR)
      find_path(CLIQUER_INCLUDE_DIR
          NAMES cliquer.h
          PATHS ${CLIQUER_DIR}
          PATH_SUFFIXES include
          NO_DEFAULT_PATH
          )

      # if we found the headers there we copy the folder to a cliquer folder in the binary dir and use that as include
      if(CLIQUER_INCLUDE_DIR)
         set(COPY_CLIQUER_HEADERS TRUE)
      endif()
   endif()

   # look for the library in the cliquer directory
   find_library(CLIQUER_LIBRARY
       NAMES cliquer
       PATHS ${CLIQUER_DIR}
       PATH_SUFFIXES lib
       NO_DEFAULT_PATH
       )

   # set variables and call handle standard args
   set(CLIQUER_LIBRARIES ${CLIQUER_LIBRARY})
   set(CLIQUER_INCLUDE_DIRS ${CLIQUER_INCLUDE_DIR})

   find_package_handle_standard_args(CLIQUER DEFAULT_MSG CLIQUER_INCLUDE_DIRS CLIQUER_LIBRARIES)

   if(CLIQUER_FOUND AND COPY_CLIQUER_HEADERS)
      file(GLOB cliquer_headers ${CLIQUER_INCLUDE_DIR}/*.h)
      file(COPY ${cliquer_headers} DESTINATION ${CMAKE_BINARY_DIR}/cliquer)
      set(CLIQUER_INCLUDE_DIR ${CMAKE_BINARY_DIR} CACHE PATH "Include path for cliquer headers" FORCE)
      set(CLIQUER_INCLUDE_DIRS ${CLIQUER_INCLUDE_DIR})
   endif()
endif()

# if cliquer is not already found by the code above we look for it including system directories
if(NOT CLIQUER_FOUND)
   find_path(CLIQUER_INCLUDE_DIR
       NAMES cliquer/cliquer.h
       PATH_SUFFIXES include)

   find_library(CLIQUER_LIBRARY
       NAMES cliquer
       PATH_SUFFIXES lib)

   set(CLIQUER_LIBRARIES ${CLIQUER_LIBRARY})
   set(CLIQUER_INCLUDE_DIRS ${CLIQUER_INCLUDE_DIR})

   find_package_handle_standard_args(CLIQUER DEFAULT_MSG CLIQUER_INCLUDE_DIRS CLIQUER_LIBRARIES)
endif()

