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
set(DEBUG_IDRC_CONDITION_AREA ON)
set(DEBUG_IDRC_CONDITION_CONNECTIVITY ON)
set(DEBUG_IDRC_CONDITION_EOL ON)
set(DEBUG_IDRC_CONDITION_JOG ON)
set(DEBUG_IDRC_CONDITION_NOTCH ON)
set(DEBUG_IDRC_CONDITION_SPACING_TABLE ON)
set(DEBUG_IDRC_CONDITION_STEP ON)
set(DEBUG_IDRC_CONDITION ON)

# debug engine
set(DEBUG_IDRC_ENGINE ON)
set(DEBUG_IDRC_ENGINE_INIT ON)
set(DEBUG_IDRC_ENGINE_SCANLINE ON)
set(DEBUG_IDRC_ENGINE_CHECK ON)
set(DEBUG_IDRC_ENGINE_MANAGER ON)

# debug tech rules
set(DEBUG_IDRC_RULE ON)

# violation
set(DEBUG_IDRC_VIOLATION ON)

# debug api
set(DEBUG_IDRC_API ON)

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
