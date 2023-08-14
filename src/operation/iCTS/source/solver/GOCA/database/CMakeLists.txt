if(DEBUG_ICTS_GOCA_DATABASE)
  message(STATUS "CTS: DEBUG_ICTS_GOCA_DATABASE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_GOCA_DATABASE")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(
  icts_goca_database
  ${ICTS_SOLVER}/GOCA/database/Pin.cc ${ICTS_SOLVER}/GOCA/database/Inst.cc
  ${ICTS_SOLVER}/GOCA/database/Net.cc)

target_link_libraries(icts_goca_database PUBLIC icts_api icts_data_manager)

target_include_directories(icts_goca_database
                           PUBLIC ${ICTS_SOLVER}/GOCA/database)
