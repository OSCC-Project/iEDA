cmake_minimum_required(VERSION 3.3)

set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELEASE}")

project(UG)

# todo: add version
set(UG_VERSION_MAJOR 1)
set(UG_VERSION_MINOR 0)
set(UG_VERSION_PATCH 0)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
   add_compile_options(-Wno-psabi)
endif()

if(WIN32)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(NO_SIGACTION)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# path to e.g. findGMP module
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# make 'Release' the default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# can also change to 11, but scip and soplex use 14
set(CMAKE_CXX_STANDARD 14)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(SCIP REQUIRED HINTS ${SCIP_DIR})
include_directories(${SCIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)

# create a target for updating the current git hash
file(WRITE ${CMAKE_BINARY_DIR}/ug_update_githash.cmake "
find_program(GIT git)
if(EXISTS \${DST})
   file(STRINGS \${DST} GITHASH_OLD)
   string(REGEX REPLACE \"#define UG_GITHASH \\\"(.*)\\\"\" \"\\\\1\" GITHASH_OLD \${GITHASH_OLD})
endif()
if((GIT) AND (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git))
   execute_process(
      COMMAND \${GIT} describe --always --dirty
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE GITHASH OUTPUT_STRIP_TRAILING_WHITESPACE)
   string(REGEX REPLACE \"^.*-g\" \"\" GITHASH \${GITHASH})
   if(NOT \${GITHASH} STREQUAL \"\${GITHASH_OLD}\")
      file(WRITE \${DST} \"#define UG_GITHASH \\\"\${GITHASH}\\\"\n\")
   endif()
else()
   set(GITHASH \${GITHASH_OLD})
   if(NOT GITHASH)
      message(STATUS \"Compiling without git information\")
      set(GITHASH \"NoGitInfo\")
   endif()
   file(WRITE \${DST} \"#define UG_GITHASH \\\"\${GITHASH}\\\"\n\")
endif()
message(STATUS \"Git hash: \" \${GITHASH})
")
add_custom_target(ug_update_githash
   COMMAND ${CMAKE_COMMAND}
           -DDST=${PROJECT_SOURCE_DIR}/src/ug/githash.cpp
           -P ${CMAKE_BINARY_DIR}/ug_update_githash.cmake)

# maybe set a variable with common sources

# todo set sources for FiberSCIP
set(fiberscip_src
   src/ug/uggithash.h
   src/ug/uggithash.cpp
   src/ug/paraTimerTh.h
   src/ug/paraTimer.h
   src/ug/paraTimeLimitMonitorTh.h
   src/ug/paraTagDef.h
   src/ug/paraSysTimer.h
   src/ug/paraSysTimer.cpp
   src/ug/paraSolverTerminationState.h
   src/ug/paraSolverState.h
   src/ug/paraSolverPool.h
   src/ug/paraSolver.h
   src/ug/paraSolver.cpp
   src/ug/paraSolution.h
   src/ug/paraRacingRampUpParamSet.h
   src/ug/paraParamSetTh.h
   src/ug/paraParamSet.h
   src/ug/paraParamSet.cpp
   src/ug/paraLoadCoordinatorTerminationState.h
   src/ug/paraLoadCoordinator.h
   src/ug/paraLoadCoordinator.cpp
   src/ug/paraInstance.h
   src/ug/paraInitiator.h
   src/ug/paraInitialStat.h
   src/ug/paraDiffSubproblem.h
   src/ug/paraDeterministicTimer.h
   src/ug/paraDef.h
   src/ug/paraCommCPP11.h
   src/ug/paraCommCPP11.cpp
   src/ug/paraComm.h
   src/ug/paraCalculationState.h
   src/ug/gzstream.h
   src/ug/gzstream.cpp
   src/ug_bb/bbParaTagDef.h
   src/ug_bb/bbParaSolverTerminationStateTh.h
   src/ug_bb/bbParaSolverTerminationStateTh.cpp
   src/ug_bb/bbParaSolverTerminationState.h
   src/ug_bb/bbParaSolverTerminationState.cpp
   src/ug_bb/bbParaSolverStateTh.h
   src/ug_bb/bbParaSolverStateTh.cpp
   src/ug_bb/bbParaSolverState.h
   src/ug_bb/bbParaSolverPool.h
   src/ug_bb/bbParaSolverPool.cpp
   src/ug_bb/bbParaSolver.h
   src/ug_bb/bbParaSolver.cpp
   src/ug_bb/bbParaSolution.h
   src/ug_bb/bbParaRacingRampUpParamSet.h
   src/ug_bb/bbParaParamSet.h
   src/ug_bb/bbParaParamSet.cpp
   src/ug_bb/bbParaNodesMerger.h
   src/ug_bb/bbParaNodesMerger.cpp
   src/ug_bb/bbParaNodeTh.h
   src/ug_bb/bbParaNodeTh.cpp
   src/ug_bb/bbParaNodePool.h
   src/ug_bb/bbParaNode.h
   src/ug_bb/bbParaNode.cpp
   src/ug_bb/bbParaLoadCoordinatorTerminationState.h
   src/ug_bb/bbParaLoadCoordinatorTerminationState.cpp
   src/ug_bb/bbParaLoadCoordinator.h
   src/ug_bb/bbParaLoadCoordinator.cpp
   src/ug_bb/bbParaInstance.h
   src/ug_bb/bbParaInitiator.h
   src/ug_bb/bbParaDiffSubproblem.h
   src/ug_bb/bbParaComm.h
   src/ug_bb/bbParaCommCPP11.h
   src/ug_bb/bbParaCommCPP11.cpp
   src/ug_bb/bbParaCalculationStateTh.h
   src/ug_bb/bbParaCalculationStateTh.cpp
   src/ug_bb/bbParaCalculationState.h
   src/ug_scip/scipUserPlugins.h
   src/ug_scip/scipParaTagDef.h
   src/ug_scip/scipParaSolver.h
   src/ug_scip/scipParaSolver.cpp
   src/ug_scip/scipParaSolutionTh.h
   src/ug_scip/scipParaSolutionTh.cpp
   src/ug_scip/scipParaSolution.h
   src/ug_scip/scipParaSolution.cpp
   src/ug_scip/scipParaRacingRampUpParamSetTh.h
   src/ug_scip/scipParaRacingRampUpParamSetTh.cpp
   src/ug_scip/scipParaRacingRampUpParamSet.h
   src/ug_scip/scipParaRacingRampUpParamSet.cpp
   src/ug_scip/scipParaParamSet.h
   src/ug_scip/scipParaParamSet.cpp
   src/ug_scip/scipParaObjProp.h
   src/ug_scip/scipParaObjNodesel.h
   src/ug_scip/scipParaObjNodesel.cpp
   src/ug_scip/scipParaObjMessageHdlr.h
   src/ug_scip/scipParaObjMessageHdlr.cpp
   src/ug_scip/scipParaObjLimitUpdator.h
   src/ug_scip/scipParaObjLimitUpdator.cpp
   src/ug_scip/scipParaObjCommPointHdlr.h
   src/ug_scip/scipParaObjCommPointHdlr.cpp
   src/ug_scip/scipParaObjBranchRule.h
   src/ug_scip/scipParaObjBranchRule.cpp
   src/ug_scip/scipParaObjSelfSplitNodesel.h
   src/ug_scip/scipParaObjSelfSplitNodesel.cpp
   src/ug_scip/scipParaInterruptMsgMonitor.h
   src/ug_scip/scipParaInterruptMsgMonitor.cpp
   src/ug_scip/scipParaLoadCoordinator.h
   src/ug_scip/scipParaLoadCoordinator.cpp
   src/ug_scip/scipParaInstanceTh.h
   src/ug_scip/scipParaInstanceTh.cpp
   src/ug_scip/scipParaInstance.h
   src/ug_scip/scipParaInitiator.h
   src/ug_scip/scipParaInitiator.cpp
   src/ug_scip/scipParaInitialStatTh.h
   src/ug_scip/scipParaInitialStatTh.cpp
   src/ug_scip/scipParaInitialStat.h
   src/ug_scip/scipParaInitialStat.cpp
   src/ug_scip/scipParaDiffSubproblemTh.h
   src/ug_scip/scipParaDiffSubproblem.h
   src/ug_scip/scipParaDiffSubproblem.cpp
   src/ug_scip/scipParaDeterministicTimer.h
   src/ug_scip/scipParaCommTh.h
   src/ug_scip/scipParaCommTh.cpp
   src/ug_scip/scipParaComm.h
   src/ug_scip/scipDiffParamSetTh.h
   src/ug_scip/scipDiffParamSetTh.cpp
   src/ug_scip/scipDiffParamSet.h
   src/ug_scip/scipDiffParamSet.cpp
   src/ug_scip/scipDefaultUserPlugins.cpp
   src/ug_scip/fscip.cpp
   )

# todo set sources for ParaSCIP
set(parascip_src
   src/ug/uggithash.h
   src/ug/uggithash.cpp
   src/ug/paraTimerMpi.h
   src/ug/paraTimerMpi.cpp
   src/ug/paraTagDef.h
   src/ug/paraSysTimer.h
   src/ug/paraSysTimer.cpp
   src/ug/paraSolverTerminationState.h
   src/ug/paraSolverState.h
   src/ug/paraSolverPool.h
   src/ug/paraSolver.h
   src/ug/paraSolver.cpp
   src/ug/paraSolution.h
   src/ug/paraRacingRampUpParamSet.h
   src/ug/paraParamSetMpi.h
   src/ug/paraParamSetMpi.cpp
   src/ug/paraParamSet.h
   src/ug/paraParamSet.cpp
   src/ug/paraLoadCoordinatorTerminationState.h
   src/ug/paraLoadCoordinator.h
   src/ug/paraLoadCoordinator.cpp
   src/ug/paraInstance.h
   src/ug/paraInitiator.h
   src/ug/paraInitialStat.h
   src/ug/paraDiffSubproblem.h
   src/ug/paraDeterministicTimer.h
   src/ug/paraDef.h
   src/ug/paraCommMpi.h
   src/ug/paraCommMpi.cpp
   src/ug/paraComm.h
   src/ug/paraCalculationState.h
   src/ug/gzstream.h
   src/ug/gzstream.cpp
   src/ug_bb/bbParaTagDef.h
   src/ug_bb/bbParaSolverTerminationStateMpi.h
   src/ug_bb/bbParaSolverTerminationStateMpi.cpp
   src/ug_bb/bbParaSolverTerminationState.h
   src/ug_bb/bbParaSolverTerminationState.cpp
   src/ug_bb/bbParaSolverStateMpi.h
   src/ug_bb/bbParaSolverStateMpi.cpp
   src/ug_bb/bbParaSolverState.h
   src/ug_bb/bbParaSolverPool.h
   src/ug_bb/bbParaSolverPool.cpp
   src/ug_bb/bbParaSolver.h
   src/ug_bb/bbParaSolver.cpp
   src/ug_bb/bbParaSolution.h
   src/ug_bb/bbParaRacingRampUpParamSet.h
   src/ug_bb/bbParaParamSet.h
   src/ug_bb/bbParaParamSet.cpp
   src/ug_bb/bbParaNodeMpi.h
   src/ug_bb/bbParaNodeMpi.cpp
   src/ug_bb/bbParaNodePool.h
   src/ug_bb/bbParaNode.h
   src/ug_bb/bbParaNode.cpp
   src/ug_bb/bbParaNodesMerger.h
   src/ug_bb/bbParaNodesMerger.cpp
   src/ug_bb/bbParaLoadCoordinatorTerminationState.h
   src/ug_bb/bbParaLoadCoordinatorTerminationState.cpp
   src/ug_bb/bbParaLoadCoordinator.h
   src/ug_bb/bbParaLoadCoordinator.cpp
   src/ug_bb/bbParaInstance.h
   src/ug_bb/bbParaInitiator.h
   src/ug_bb/bbParaDiffSubproblem.h
   src/ug_bb/bbParaComm.h
   src/ug_bb/bbParaCommMpi.h
   src/ug_bb/bbParaCommMpi.cpp
   src/ug_bb/bbParaCalculationStateMpi.h
   src/ug_bb/bbParaCalculationStateMpi.cpp
   src/ug_bb/bbParaCalculationState.h
   src/ug_scip/scipUserPlugins.h
   src/ug_scip/scipParaTagDef.h
   src/ug_scip/scipParaSolver.h
   src/ug_scip/scipParaSolver.cpp
   src/ug_scip/scipParaSolutionMpi.h
   src/ug_scip/scipParaSolutionMpi.cpp
   src/ug_scip/scipParaSolution.h
   src/ug_scip/scipParaSolution.cpp
   src/ug_scip/scipParaRacingRampUpParamSetMpi.h
   src/ug_scip/scipParaRacingRampUpParamSetMpi.cpp
   src/ug_scip/scipParaRacingRampUpParamSet.h
   src/ug_scip/scipParaRacingRampUpParamSet.cpp
   src/ug_scip/scipParaParamSet.h
   src/ug_scip/scipParaParamSet.cpp
   src/ug_scip/scipParaObjProp.h
   src/ug_scip/scipParaObjNodesel.h
   src/ug_scip/scipParaObjNodesel.cpp
   src/ug_scip/scipParaObjMessageHdlr.h
   src/ug_scip/scipParaObjMessageHdlr.cpp
   src/ug_scip/scipParaObjLimitUpdator.h
   src/ug_scip/scipParaObjLimitUpdator.cpp
   src/ug_scip/scipParaObjCommPointHdlr.h
   src/ug_scip/scipParaObjCommPointHdlr.cpp
   src/ug_scip/scipParaObjBranchRule.h
   src/ug_scip/scipParaObjBranchRule.cpp
   src/ug_scip/scipParaObjSelfSplitNodesel.h
   src/ug_scip/scipParaObjSelfSplitNodesel.cpp
   src/ug_scip/scipParaInterruptMsgMonitor.h
   src/ug_scip/scipParaInterruptMsgMonitor.cpp
   src/ug_scip/scipParaLoadCoordinator.h
   src/ug_scip/scipParaLoadCoordinator.cpp
   src/ug_scip/scipParaInstanceMpi.h
   src/ug_scip/scipParaInstanceMpi.cpp
   src/ug_scip/scipParaInstance.h
   src/ug_scip/scipParaInstance.cpp
   src/ug_scip/scipParaInitiator.h
   src/ug_scip/scipParaInitiator.cpp
   src/ug_scip/scipParaInitialStatMpi.h
   src/ug_scip/scipParaInitialStatMpi.cpp
   src/ug_scip/scipParaInitialStat.h
   src/ug_scip/scipParaInitialStat.cpp
   src/ug_scip/scipParaDiffSubproblemMpi.h
   src/ug_scip/scipParaDiffSubproblemMpi.cpp
   src/ug_scip/scipParaDiffSubproblem.h
   src/ug_scip/scipParaDiffSubproblem.cpp
   src/ug_scip/scipParaDeterministicTimer.h
   src/ug_scip/scipParaCommMpi.h
   src/ug_scip/scipParaCommMpi.cpp
   src/ug_scip/scipParaComm.h
   src/ug_scip/scipDiffParamSetMpi.h
   src/ug_scip/scipDiffParamSetMpi.cpp
   src/ug_scip/scipDiffParamSet.h
   src/ug_scip/scipDiffParamSet.cpp
   src/ug_scip/scipDefaultUserPlugins.cpp
   src/ug_scip/parascip.cpp
   )

add_executable(fscip ${fiberscip_src})
add_dependencies(fscip ug_update_githash)
find_package( Threads REQUIRED )
target_link_libraries(fscip ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(fscip ${SCIP_LIBRARIES})

find_package(ZLIB)

if(ZLIB_FOUND)
   target_compile_definitions(fscip PRIVATE UG_WITH_ZLIB)
   target_link_libraries(fscip ${ZLIB_LIBRARIES})
endif()

# todo add definitions for compiling fscip with c++11 threads
target_compile_definitions(fscip PRIVATE
   _COMM_CPP11
   _SCIP_SOLVER
   _REENTRANT)

set_target_properties(fscip PROPERTIES
   INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
   INSTALL_RPATH_USE_LINK_PATH TRUE)

# add parascip if MPI is found
find_package( MPI )
if(MPI_FOUND)
   add_executable(parascip ${parascip_src})
   add_dependencies(parascip ug_update_githash)

   if(ZLIB_FOUND)
      target_compile_definitions(parascip PRIVATE UG_WITH_ZLIB)
      target_link_libraries(parascip ${ZLIB_LIBRARIES})
   endif()

   # todo add definitions for compiling parascip with MPI
   target_compile_definitions(parascip PRIVATE
      _COMM_MPI_WORLD
      MPICH_IGNORE_CXX_SEEK
      _SCIP_SOLVER
      _MUTEX_CPP11)

   # todo does parascip need to link pthread?
   # target_link_libraries(parascip ${CMAKE_THREAD_LIBS_INIT})
   target_link_libraries(parascip ${SCIP_LIBRARIES})
   target_link_libraries(parascip ${MPI_LIBRARIES})
   target_include_directories(parascip PRIVATE ${MPI_INCLUDE_PATH})

   set_target_properties(parascip PROPERTIES
      INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
      INSTALL_RPATH_USE_LINK_PATH TRUE)

   if(MPI_COMPILE_FLAGS)
   set_target_properties(parascip PROPERTIES
      COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
   endif()

   if(MPI_LINK_FLAGS)
   set_target_properties(parascip PROPERTIES
      LINK_FLAGS "${MPI_LINK_FLAGS}")
   endif()

   # install the binary
   install(TARGETS parascip
           RUNTIME DESTINATION bin)
endif()

# install the binary
install(TARGETS fscip
        RUNTIME DESTINATION bin)
