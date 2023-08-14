#
# interface function for setting common library properties
#
function(setLibProperties targetname outputname)
    set_target_properties(${targetname} PROPERTIES
        OUTPUT_NAME ${outputname}
        MACOSX_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endfunction(setLibProperties)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set(sources
    soplex/didxset.cpp
    soplex/gzstream.cpp
    soplex/idxset.cpp
    soplex/mpsinput.cpp
    soplex/nameset.cpp
    soplex/spxdefines.cpp
    soplex/spxgithash.cpp
    soplex/spxid.cpp
    soplex/spxout.cpp
    soplex/usertimer.cpp
    soplex/wallclocktimer.cpp)

set(headers
    soplex/array.h
    soplex/basevectors.h
    soplex/classarray.h
    soplex/classset.h
    soplex/clufactor.h
    soplex/clufactor_rational.h
    soplex/cring.h
    soplex/dataarray.h
    soplex/datahashtable.h
    soplex/datakey.h
    soplex/dataset.h
    soplex/didxset.h
    soplex/dsvectorbase.h
    soplex/dsvector.h
    soplex/exceptions.h
    soplex/gzstream.h
    soplex/idlist.h
    soplex/idxset.h
    soplex/islist.h
    soplex/lpcolbase.h
    soplex/lpcol.h
    soplex/lpcolsetbase.h
    soplex/lpcolset.h
    soplex/lprowbase.h
    soplex/lprow.h
    soplex/lprowsetbase.h
    soplex/lprowset.h
    soplex/mpsinput.h
    soplex/nameset.h
    soplex/notimer.h
    soplex/random.h
    soplex/rational.h
    soplex/ratrecon.h
    soplex/slinsolver.h
    soplex/slinsolver_rational.h
    soplex/slufactor.h
    soplex/slufactor_rational.h
    soplex/solbase.h
    soplex/sol.h
    soplex/sorter.h
    soplex/spxalloc.h
    soplex/spxautopr.h
    soplex/spxbasis.h
    soplex/spxboundflippingrt.h
    soplex/spxdantzigpr.h
    soplex/spxdefaultrt.h
    soplex/spxdefines.h
    soplex/spxdevexpr.h
    soplex/spxequilisc.h
    soplex/spxfastrt.h
    soplex/spxfileio.h
    soplex/spxgeometsc.h
    soplex/spxgithash.h
    soplex/spxharrisrt.h
    soplex/spxhybridpr.h
    soplex/spxid.h
    soplex/spxleastsqsc.h
    soplex/spxlpbase.h
    soplex/spxlp.h
    soplex/spxmainsm.h
    soplex/spxout.h
    soplex/spxparmultpr.h
    soplex/spxpapilo.h
    soplex/spxpricer.h
    soplex/spxratiotester.h
    soplex/spxscaler.h
    soplex/spxsimplifier.h
    soplex/spxsolver.h
    soplex/spxstarter.h
    soplex/spxsteepexpr.h
    soplex/spxsteeppr.h
    soplex/spxsumst.h
    soplex/spxvectorst.h
    soplex/spxweightpr.h
    soplex/spxweightst.h
    soplex/ssvectorbase.h
    soplex/ssvector.h
    soplex/stablesum.h
    soplex/statistics.h
    soplex/svectorbase.h
    soplex/svector.h
    soplex/svsetbase.h
    soplex/svset.h
    soplex/timerfactory.h
    soplex/timer.h
    soplex/unitvectorbase.h
    soplex/unitvector.h
    soplex/updatevector.h
    soplex/usertimer.h
    soplex/validation.h
    soplex/vectorbase.h
    soplex/vector.h
    soplex/dvector.h
    soplex/wallclocktimer.h
    )

file(GLOB headers
    ${headers}
    soplex/*.hpp
)

if(MSVC)
   set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# create soplex library with pic
add_library(libsoplex-pic STATIC ${sources})
if(MSVC)
    setLibProperties(libsoplex-pic "libsoplex-pic")
else()
   setLibProperties(libsoplex-pic "soplex-pic")
endif()

set_target_properties(libsoplex-pic PROPERTIES
    POSITION_INDEPENDENT_CODE on
    VERSION ${SOPLEX_VERSION_MAJOR}.${SOPLEX_VERSION_MINOR}.${SOPLEX_VERSION_PATCH}.${SOPLEX_VERSION_SUB}
    SOVERSION ${SOPLEX_VERSION_MAJOR}.${SOPLEX_VERSION_MINOR})
add_dependencies(libsoplex-pic soplex_update_githash)
target_include_directories(libsoplex-pic PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(libsoplex-pic ${libs})

# create soplex library without pic
add_library(libsoplex STATIC ${sources})
if(MSVC)
    setLibProperties(libsoplex "libsoplex")
else()
   setLibProperties(libsoplex "soplex")
endif()

set_target_properties(libsoplex PROPERTIES
    VERSION ${SOPLEX_VERSION_MAJOR}.${SOPLEX_VERSION_MINOR}.${SOPLEX_VERSION_PATCH}.${SOPLEX_VERSION_SUB}
    SOVERSION ${SOPLEX_VERSION_MAJOR}.${SOPLEX_VERSION_MINOR})
add_dependencies(libsoplex soplex_update_githash)
target_include_directories(libsoplex PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(libsoplex ${libs})

# create C shared library
add_library(libsoplexshared SHARED soplex_interface.cpp ${sources})
if(MSVC)
    setLibProperties(libsoplexshared "libsoplexshared")
else()
   setLibProperties(libsoplexshared "soplexshared")
endif()

set_target_properties(libsoplexshared PROPERTIES
    VERSION ${SOPLEX_VERSION_MAJOR}.${SOPLEX_VERSION_MINOR}.${SOPLEX_VERSION_PATCH}.${SOPLEX_VERSION_SUB}
    SOVERSION ${SOPLEX_VERSION_MAJOR}.${SOPLEX_VERSION_MINOR})
add_dependencies(libsoplexshared soplex_update_githash libsoplex)
if(MSVC)
    target_compile_options(libsoplexshared PUBLIC "/Zc:__cplusplus")
    setLibProperties(libsoplexshared "libsoplexshared")
endif()
target_include_directories(libsoplexshared PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(libsoplexshared libsoplex ${libs})
set_target_properties(libsoplexshared PROPERTIES CXX_VISIBILITY_PRESET default)

# create soplex binary using library without pic
add_executable(soplex soplexmain.cpp)
target_link_libraries(soplex LINK_PUBLIC libsoplex ${Boost_LIBRARIES})

if(CMAKE_BUILD_TYPE EQUAL "Debug")
   find_package(Sanitizers)
   add_sanitizers(soplex)
endif()

add_executable(example EXCLUDE_FROM_ALL example.cpp)
target_link_libraries(example libsoplex)

# set the install rpath to the installed destination
set_target_properties(soplex PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# install the header files of soplex
install(FILES ${headers} ${PROJECT_BINARY_DIR}/soplex/config.h DESTINATION include/soplex)
install(FILES soplex.h soplex.hpp soplex_interface.h DESTINATION include)

# install the binary and the library to appropriate lcoations and add them to an export group
install(TARGETS soplex libsoplex libsoplex-pic libsoplexshared EXPORT soplex-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Add library targets to the build-tree export set
export(TARGETS libsoplex libsoplex-pic libsoplexshared
  FILE "${CMAKE_BINARY_DIR}/soplex-targets.cmake")

#configure the config file for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}")
configure_file(${PROJECT_SOURCE_DIR}/soplex-config.cmake.in
  "${CMAKE_BINARY_DIR}/soplex-config.cmake" @ONLY)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
${CMAKE_BINARY_DIR}/soplex-config-version.cmake
    VERSION ${SOPLEX_VERSION_MAJOR}.${SOPLEX_VERSION_MINOR}.${SOPLEX_VERSION_PATCH}
    COMPATIBILITY SameMajorVersion
    )


#configure the config file for the install
set(CONF_INCLUDE_DIRS "\${CMAKE_CURRENT_LIST_DIR}/../../../include")
configure_file(${PROJECT_SOURCE_DIR}/soplex-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/soplex-config.cmake" @ONLY)

# install the targets of the soplex export group and the config file so that other projects
# can link easily against soplex
install(EXPORT soplex-targets FILE soplex-targets.cmake DESTINATION lib/cmake/soplex)
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/soplex-config.cmake"
	           ${CMAKE_BINARY_DIR}/soplex-config-version.cmake
	     DESTINATION lib/cmake/soplex)
