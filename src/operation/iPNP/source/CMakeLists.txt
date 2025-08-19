add_subdirectory(config)
add_subdirectory(data_manager)
add_subdirectory(module)

add_library(pnp PNP.cpp)

target_link_libraries(pnp
  pnp-evaluator
  pnp-optimizer
  pnp-synthesis
  pnp-data
  pnp-config

  log
  usage
  pthread
  stdc++fs
)

if(BUILD_PNP_EXECUTE)
  target_link_libraries(pnp
    pnp-cmd
)
endif()

target_include_directories(pnp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
