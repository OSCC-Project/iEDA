# 3.5 is actually available almost everywhere, but this a good minimum
cmake_minimum_required(VERSION 3.11)

# enable MSVC_RUNTIME_LIBRARY target property
# see https://cmake.org/cmake/help/latest/policy/CMP0091.html
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(YAML_CPP VERSION 0.6.3 LANGUAGES CXX)

option(YAML_CPP_BUILD_CONTRIB "Enable yaml-cpp contrib in library" ON)
option(YAML_CPP_BUILD_TOOLS "Enable parse tools" ON)
option(YAML_BUILD_SHARED_LIBS "Build yaml-cpp shared library" ${BUILD_SHARED_LIBS})


if (YAML_BUILD_SHARED_LIBS)
  set(yaml-cpp-type SHARED)
  set(yaml-cpp-label-postfix "shared")
else()
  set(yaml-cpp-type STATIC)
  set(yaml-cpp-label-postfix "static")
endif()

set(contrib-pattern "src/contrib/*.cpp")
set(src-pattern "src/*.cpp")


file(GLOB yaml-cpp-contrib-sources ${contrib-pattern})
file(GLOB yaml-cpp-sources ${src-pattern})


add_library(yaml-cpp ${yaml-cpp-type} "")
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)


set_property(TARGET yaml-cpp
  PROPERTY
    CXX_STANDARD_REQUIRED ON)

target_include_directories(yaml-cpp
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

target_sources(yaml-cpp
  PRIVATE
    $<$<BOOL:${YAML_CPP_BUILD_CONTRIB}>:${yaml-cpp-contrib-sources}>
    ${yaml-cpp-sources})

if (NOT DEFINED CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX "d")
endif()

configure_file(yaml-cpp.pc.in yaml-cpp.pc @ONLY)


