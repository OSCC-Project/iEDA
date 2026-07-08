set(ICTS_MODULE_CHAR_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/Characterization.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/builder/CharBuilder.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/builder/CharBuilderImpl.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/builder/CharBuildOrchestrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/builder/CharFeasibilityChecker.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/builder/CharSetupConfigurator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/builder/CharTopologyPlanner.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/circuit/CharCircuitBuilder.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pattern/CharPatternEnumerator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pattern/CharPatternStorage.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sampling/CharStaSampler.cc
)

add_library(icts_source_module_characterization_headers INTERFACE)

target_link_libraries(
  icts_source_module_characterization_headers
  INTERFACE
  icts_source_database_characterization
  icts_source_database_routing
)

target_include_directories(
  icts_source_module_characterization_headers
  INTERFACE
  ${ICTS_MODULE}
)

add_library(icts_source_module_characterization ${ICTS_MODULE_CHAR_SRC})

target_link_libraries(
  icts_source_module_characterization
  PUBLIC
  icts_source_module_characterization_headers
  icts_source_database_characterization
  icts_source_database_routing
  PRIVATE
  icts_source_database_adapter_fast_sta
  icts_source_utils_logger
  log
)

if(DEBUG_ICTS_SOURCE_MODULE_CHARACTERIZATION)
  if(MSVC)
    target_compile_options(icts_source_module_characterization PRIVATE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_module_characterization PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
