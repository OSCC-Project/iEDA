include(${HOME_CMAKE}/operation/idb.cmake)
include(${HOME_CMAKE}/operation/iplf.cmake)

INCLUDE_DIRECTORIES(
    SYSTEM
    ## Third Party
    ${HOME_THIRDPARTY}/json
    ## Boost
    ${Boost_INCLUDE_DIRS} 
)


# debug condition
set(DEBUG_IDRC_CONDITION_AREA OFF)
set(DEBUG_IDRC_CONDITION_CONNECTIVITY OFF)
set(DEBUG_IDRC_CONDITION_EOL OFF)
set(DEBUG_IDRC_CONDITION_JOG OFF)
set(DEBUG_IDRC_CONDITION_NOTCH OFF)
set(DEBUG_IDRC_CONDITION_SPACING_TABLE OFF)
set(DEBUG_IDRC_CONDITION_STEP OFF)
set(DEBUG_IDRC_CONDITION OFF)

# debug engine
set(DEBUG_IDRC_ENGINE OFF)
set(DEBUG_IDRC_ENGINE_INIT OFF)
set(DEBUG_IDRC_ENGINE_SCANLINE OFF)
set(DEBUG_IDRC_ENGINE_CHECK OFF)
set(DEBUG_IDRC_ENGINE_MANAGER OFF)

# debug tech rules
set(DEBUG_IDRC_RULE OFF)

# violation
set(DEBUG_IDRC_VIOLATION OFF)

# debug api
set(DEBUG_IDRC_API OFF)

if(DEBUG_IDRC_CONDITION_AREA)
  add_compile_definitions(DEBUG_IDRC_CONDITION_AREA)
endif()
if(DEBUG_IDRC_CONDITION_CONNECTIVITY)
  add_compile_definitions(DEBUG_IDRC_CONDITION_CONNECTIVITY)
endif()
if(DEBUG_IDRC_CONDITION_EOL)
  add_compile_definitions(DEBUG_IDRC_CONDITION_EOL)
endif()
if(DEBUG_IDRC_CONDITION_JOG)
  add_compile_definitions(DEBUG_IDRC_CONDITION_JOG)
endif()
if(DEBUG_IDRC_CONDITION_NOTCH)
  add_compile_definitions(DEBUG_IDRC_CONDITION_NOTCH)
endif()
if(DEBUG_IDRC_CONDITION_SPACING_TABLE)
  add_compile_definitions(DEBUG_IDRC_CONDITION_SPACING_TABLE)
endif()
if(DEBUG_IDRC_CONDITION_STEP)
  add_compile_definitions(DEBUG_IDRC_CONDITION_STEP)
endif()
if(DEBUG_IDRC_CONDITION)
  add_compile_definitions(DEBUG_IDRC_CONDITION)
endif()
if(DEBUG_IDRC_ENGINE)
  add_compile_definitions(DEBUG_IDRC_ENGINE)
endif()
if(DEBUG_IDRC_ENGINE_INIT)
  add_compile_definitions(DEBUG_IDRC_ENGINE_INIT)
endif()
if(DEBUG_IDRC_ENGINE_SCANLINE)
  add_compile_definitions(DEBUG_IDRC_ENGINE_SCANLINE)
endif()
if(DEBUG_IDRC_ENGINE_CHECK)
  add_compile_definitions(DEBUG_IDRC_ENGINE_CHECK)
endif()
if(DEBUG_IDRC_RULE)
  add_compile_definitions(DEBUG_IDRC_RULE)
endif()
if(DEBUG_IDRC_API)
  add_compile_definitions(DEBUG_IDRC_API)
endif()

add_subdirectory(api)
add_subdirectory(source)
# add_subdirectory(test)
