if(PY_MODEL)
  if(DEBUG_ICTS_PYTHON)
    message(STATUS "CTS: DEBUG_ICTS_PYTHON")
    set(CMAKE_BUILD_TYPE "Debug")
  else()
    message(STATUS "CTS: RELEASE_ICTS_PYTHON")
    set(CMAKE_BUILD_TYPE "Release")
  endif()
  add_compile_options(-w)
  add_library(icts_python ${ICTS_PYTHON}/PyModel.cpp)

  find_package(Python3 ${SYS_PYTHON3_VERSION} EXACT QUIET REQUIRED
               COMPONENTS Development)
  message(STATUS "CTS: Python3 ${Python3_INCLUDE_DIRS}")

  target_link_libraries(icts_python PUBLIC Python3::Python)

  target_include_directories(icts_python PUBLIC ${ICTS_PYTHON}/python)
endif(PY_MODEL)
