add_library(idrc_pro_api
    idrc_api.cpp
)

target_link_libraries(idrc_pro_api
    PUBLIC
        idrc_pro_src
        idrc_pro_config
        idrc_pro_data
        idrc_pro_dm
        idrc_pro_violation
        idrc_pro_rule_stratagy
        idrc_pro_rule_builder
        idm
)
if(USE_PROFILER)
  target_link_libraries(idrc_pro_api
      PUBLIC
          ${LIBPROFILER_LIBRARY}
  )
endif(USE_PROFILER)

target_include_directories(idrc_pro_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)

if(DEBUG_IDRC_API)
  message(STATUS "iDRC: DEBUG_IDRC_API")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "iDRC: RELEASE_IDRC_API")
  set(CMAKE_BUILD_TYPE "Release")
endif()