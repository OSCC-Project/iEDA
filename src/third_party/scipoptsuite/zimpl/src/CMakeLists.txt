bison_target(MMLPARSE2 zimpl/mmlparse2.y ${CMAKE_CURRENT_BINARY_DIR}/zimpl/mmlparse2.c COMPILE_FLAGS -t)
flex_target(MMLSCAN zimpl/mmlscan.l ${CMAKE_CURRENT_BINARY_DIR}/zimpl/mmlscan.c)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/zimpl)

# add include directory for bison header
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# for zimpl headers
include_directories(zimpl)

set(libsources
    zimpl/blkmem.c
    zimpl/bound.c
    zimpl/code.c
    zimpl/conname.c
    zimpl/define.c
    zimpl/elem.c
    zimpl/entry.c
    zimpl/hash.c
    zimpl/heap.c
    zimpl/idxset.c
    zimpl/inst.c
    zimpl/iread.c
    zimpl/list.c
    zimpl/load.c
    zimpl/local.c
    zimpl/metaio.c
    ${BISON_MMLPARSE2_OUTPUTS}
    ${FLEX_MMLSCAN_OUTPUTS}
    zimpl/mono.c
    zimpl/mshell.c
    zimpl/prog.c
    zimpl/random.c
    zimpl/rdefpar.c
    zimpl/source.c
    zimpl/setempty.c
    zimpl/setpseudo.c
    zimpl/setlist.c
    zimpl/setrange.c
    zimpl/setprod.c
    zimpl/setmulti.c
    zimpl/set4.c
    zimpl/stmt.c
    zimpl/stkchk.c
    zimpl/strstore2.c
    zimpl/symbol.c
    zimpl/term2.c
    zimpl/tuple.c
    zimpl/vinst.c
    zimpl/zimpllib.c)

if(GMP_FOUND)
    set(libsources ${libsources} zimpl/gmpmisc.c zimpl/numbgmp.c)
else()
    set(libsources ${libsources} zimpl/numbdbl.c)
endif()

set(binsources
    zimpl/zimpl.c
    zimpl/xlpglue.c
    zimpl/zlpglue.c
    zimpl/ratlpstore.c
    zimpl/ratlpfwrite.c
    zimpl/ratmpswrite.c
    zimpl/ratmstwrite.c
    zimpl/ratordwrite.c
    zimpl/ratqbowrite.c
    zimpl/ratpresolve.c)

set(headers
    zimpl/attribute.h
    zimpl/blkmem.h
    zimpl/bound.h
    zimpl/code.h
    zimpl/conname.h
    zimpl/define.h
    zimpl/elem.h
    zimpl/entry.h
    zimpl/gmpmisc.h
    zimpl/hash.h
    zimpl/heap.h
    zimpl/idxset.h
    zimpl/inst.h
    zimpl/lint.h
    zimpl/list.h
    zimpl/local.h
    zimpl/metaio.h
    zimpl/mme.h
    ${BISON_MMLPARSE2_OUTPUT_HEADER}
    zimpl/mono.h
    zimpl/mshell.h
    zimpl/numb.h
    zimpl/prog.h
    zimpl/random.h
    zimpl/ratlp.h
    zimpl/ratlpstore.h
    zimpl/ratlptypes.h
    zimpl/rdefpar.h
    zimpl/set4.h
    zimpl/set.h
    zimpl/stkchk.h
    zimpl/stmt.h
    zimpl/strstore.h
    zimpl/symbol.h
    zimpl/term.h
    zimpl/tuple.h
    zimpl/xlpglue.h
    zimpl/zimpllib.h
    zimpl/zlpglue.h)

include_directories(${PROJECT_SOURCE_DIR}/src)

if(MSVC)
   set(headers ${headers} WIN/unistd.h)
   set(libsources ${libsources} WIN/getopt.c)
   include_directories(${PROJECT_SOURCE_DIR}/src/WIN)
endif()

#create zimpl with pic
add_library(libzimpl-pic STATIC ${libsources})
set_target_properties(libzimpl-pic PROPERTIES POSITION_INDEPENDENT_CODE on)
target_link_libraries(libzimpl-pic ${libs})

#create zimpl without pic
add_library(libzimpl STATIC ${libsources})
target_link_libraries(libzimpl ${libs})

#create zimpl binary
add_executable(zimpl ${binsources})
target_link_libraries(zimpl libzimpl ${libs})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (SANITIZE_ADDRESS OR SANITIZE_MEMORY OR SANITIZE_THREAD OR SANITIZE_UNDEFINED)
        find_package(Sanitizers)
        add_sanitizers(zimpl)
    endif()
endif()

set_target_properties(libzimpl PROPERTIES
    OUTPUT_NAME "zimpl")

set_target_properties(libzimpl-pic PROPERTIES
    OUTPUT_NAME "zimpl-pic")

# install the header files of zimpl
install(FILES ${headers} DESTINATION include/zimpl)

# install the binary and the library to appropriate locations and add them to an export group
install(TARGETS libzimpl zimpl libzimpl-pic EXPORT zimpl-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Add library targets to the build-tree export set
export(TARGETS libzimpl libzimpl-pic
  FILE "${CMAKE_BINARY_DIR}/zimpl-targets.cmake")

#configure the config file for the build tree
set(CONF_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)
if(MSVC)
   set(CONF_INCLUDE_DIRS ${CONF_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/WIN)
endif()

configure_file(${PROJECT_SOURCE_DIR}/zimpl-config.cmake.in
  "${CMAKE_BINARY_DIR}/zimpl-config.cmake" @ONLY)

#configure the config file for the install
set(CONF_INCLUDE_DIRS \${CMAKE_CURRENT_LIST_DIR}/../../../include)
if(MSVC)
   set(CONF_INCLUDE_DIRS ${CONF_INCLUDE_DIRS} \${CMAKE_CURRENT_LIST_DIR}/../../../include/WIN)
endif()

configure_file(${PROJECT_SOURCE_DIR}/zimpl-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/zimpl-config.cmake" @ONLY)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	 ${CMAKE_BINARY_DIR}/zimpl-config-version.cmake
	 VERSION ${ZIMPL_VERSION_MAJOR}.${ZIMPL_VERSION_MINOR}.${ZIMPL_VERSION_PATCH}
	 COMPATIBILITY SameMajorVersion
	 )

# install the targets of the zimpl export group and the config file so that oter projects
# can link easily against zimpl
install(EXPORT zimpl-targets FILE zimpl-targets.cmake DESTINATION lib/cmake/zimpl)
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/zimpl-config.cmake"
   "${CMAKE_BINARY_DIR}/zimpl-config-version.cmake"
   DESTINATION lib/cmake/zimpl)
