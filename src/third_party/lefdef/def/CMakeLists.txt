# DEF PARSER CMAKE Compile Settings
project(DEF_PARSER 
  LANGUAGES CXX 
)

cmake_minimum_required(VERSION 3.11)
set(DEFLIB_HOME ${CMAKE_CURRENT_SOURCE_DIR})

# BISON and ZLIB is required to compile DEF/LEF Parsers.
find_package(BISON)

if(BUILD_STATIC_LIB)
  # This is a workaround on cmake 3.22.1, should ensure static libraries are used
  # Note: There is a better approach in newer versions of CMake:
  # set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "Use static zlib" FORCE)

  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

  find_package(ZLIB REQUIRED)

  set(CMAKE_FIND_LIBRARY_SUFFIXES ${ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

  if(ZLIB_LIBRARIES MATCHES "\\.so(\\.|$)")
    message(FATAL_ERROR "Found dynamic zlib, but static required: ${ZLIB_LIBRARIES}")
  endif()

  message(STATUS "Linking static zlib: ${ZLIB_LIBRARIES}")
else()
  find_package(ZLIB REQUIRED)
endif()

############################################################
# Create a library for DEF
############################################################
set( DEFLIB_DEF_SRC
  def/defiAlias.cpp
  def/defiAssertion.cpp
  def/defiBlockage.cpp
  def/defiComponent.cpp
  def/defiDebug.cpp
  def/defiFill.cpp
  def/defiFPC.cpp
  def/defiGroup.cpp
  def/defiIOTiming.cpp
  def/defiMisc.cpp
  def/defiNet.cpp
  def/defiNonDefault.cpp
  def/defiPartition.cpp
  def/defiPath.cpp
  def/defiPinCap.cpp
  def/defiPinProp.cpp
  def/defiProp.cpp
  def/defiPropType.cpp
  def/defiRegion.cpp
  def/defiRowTrack.cpp
  def/defiScanchain.cpp
  def/defiSite.cpp
  def/defiSlot.cpp
  def/defiTimingDisable.cpp
  def/defiUtil.cpp
  def/defiVia.cpp
  def/def_keywords.cpp
  def/defrCallbacks.cpp
  def/defrData.cpp
  def/defrReader.cpp
  def/defrSettings.cpp
  def/defwWriterCalls.cpp
  def/defwWriter.cpp
  def/def.tab.cpp
  )

set( DEFLIB_DEFZLIB_SRC 
  defzlib/defzlib.cpp
  )

set (DEFLIB_HEADERS
  def/def.tab.h
  def/lex.h
  def/defiAlias.hpp
  def/defiAssertion.hpp
  def/defiBlockage.hpp
  def/defiComponent.hpp
  def/defiDebug.hpp
  def/defiDefs.hpp
  def/defiFill.hpp
  def/defiFPC.hpp
  def/defiGroup.hpp
  def/defiIOTiming.hpp
  def/defiKRDefs.hpp
  def/defiMisc.hpp
  def/defiNet.hpp
  def/defiNonDefault.hpp
  def/defiPartition.hpp
  def/defiPath.hpp
  def/defiPinCap.hpp
  def/defiPinProp.hpp
  def/defiProp.hpp
  def/defiPropType.hpp
  def/defiRegion.hpp
  def/defiRowTrack.hpp
  def/defiScanchain.hpp
  def/defiSite.hpp
  def/defiSlot.hpp
  def/defiTimingDisable.hpp
  def/defiUser.hpp
  def/defiUtil.hpp
  def/defiVia.hpp
  def/defrCallBacks.hpp
  def/defrData.hpp
  def/defrReader.hpp
  def/defrSettings.hpp
  def/defwWriterCalls.hpp
  def/defwWriter.hpp

  defzlib/defzlib.hpp
)


# BISON dependencies
add_custom_command( 
  OUTPUT ${DEFLIB_HOME}/def/def.tab.cpp
  COMMAND ${BISON_EXECUTABLE} -v -pdefyy -d ${DEFLIB_HOME}/def/def.y 
  COMMAND mv def.tab.h ${DEFLIB_HOME}/def/def.tab.h
  COMMAND mv def.tab.c ${DEFLIB_HOME}/def/def.tab.cpp
)
add_custom_target(DefBisonTarget ALL DEPENDS def/def.tab.cpp)


add_library( def ${DEFLIB_DEF_SRC} )
add_library( defzlib ${DEFLIB_DEFZLIB_SRC} )

set_target_properties( def
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${DEFLIB_HOME}/lib
  LIBRARY_OUTPUT_DIRECTORY ${DEFLIB_HOME}/lib
  RUNTIME_OUTPUT_DIRECTORY ${DEFLIB_HOME}/lib
  # python regression requirement (gag me) -cherry
  POSITION_INDEPENDENT_CODE ON
)

set_target_properties( defzlib
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${DEFLIB_HOME}/lib
  LIBRARY_OUTPUT_DIRECTORY ${DEFLIB_HOME}/lib
  RUNTIME_OUTPUT_DIRECTORY ${DEFLIB_HOME}/lib
  # python regression requirement (gag me) -cherry
  POSITION_INDEPENDENT_CODE ON
)

# include directory settings
target_include_directories( def
  PUBLIC
  def
  )

target_include_directories( defzlib
  PUBLIC
  def
  defzlib 
  )

if (BUILD_STATIC_LIB)
  message(STATUS "Linking static zlib: ${ZLIB_LIBRARIES}")
  target_link_libraries(defzlib PRIVATE ${ZLIB_LIBRARIES})
else()
  target_link_libraries(defzlib ${ZLIB_LIBRARIES})
  # the below also works.
  # target_link_libraries(defzlib -lz)
endif()

add_compile_options(-w)

add_dependencies( def DefBisonTarget )
add_dependencies( defzlib def )

######################################################
# install

install(FILES ${DEFLIB_HEADERS} DESTINATION include)
install(TARGETS def defzlib DESTINATION lib)
