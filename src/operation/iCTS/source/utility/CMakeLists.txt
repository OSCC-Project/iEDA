if(DEBUG_ICTS_UTILITY)
  message(STATUS "CTS: DEBUG_ICTS_UTILITY")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_UTILITY")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_utility INTERFACE)

target_link_libraries(icts_utility INTERFACE icts_solver)

target_include_directories(icts_utility INTERFACE ${ICTS_UTILITY})
