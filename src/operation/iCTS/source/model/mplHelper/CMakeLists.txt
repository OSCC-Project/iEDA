if(DEBUG_ICTS_MPL_HELPER)
  message(STATUS "CTS: DEBUG_ICTS_MPL_HELPER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_MPL_HELPER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_mpl_helper INTERFACE)

if(PY_MODEL)
  target_link_libraries(icts_mpl_helper INTERFACE icts_python)

  target_include_directories(
    icts_mpl_helper INTERFACE ${ICTS_MPL_HELPER} ${ICTS_PYTHON})
endif()
