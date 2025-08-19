add_subdirectory(evaluator)
add_subdirectory(optimizer)
add_subdirectory(synthesis)

if(BUILD_PNP_EXECUTE)
  add_subdirectory(pnp-cmd)
endif()
