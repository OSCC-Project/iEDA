if(DEBUG_ICTS_SYNTHESIS)
  message(STATUS "CTS: DEBUG_ICTS_SYNTHESIS")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_SYNTHESIS")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(
  icts_synthesis
  ${ICTS_MODULE}/synthesis/Synthesis.cc
  ${ICTS_MODULE}/synthesis/data/GridGraph.cc
  ${ICTS_MODULE}/synthesis/operator/Placer.cc)

target_link_libraries(icts_synthesis PUBLIC icts_data_manager icts_module)

target_include_directories(
  icts_synthesis PUBLIC ${ICTS_MODULE}/synthesis ${ICTS_MODULE}/synthesis/data
                        ${ICTS_MODULE}/synthesis/operator)
