if(DEBUG_ICTS_ROUTER)
  message(STATUS "CTS: DEBUG_ICTS_ROUTER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_ROUTER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_router ${ICTS_MODULE}/router/Router.cc)

target_link_libraries(icts_router PUBLIC icts_data_manager icts_module
                                         icts_solver)

target_include_directories(icts_router PUBLIC ${ICTS_MODULE}/router
                                              ${ICTS_SOLVER})
