set(ICTS_MODULE_ANALYTICAL_CHARACTERIZATION_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyticalCharacterization.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyticalFit.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyticalModel.cc
)

add_library(icts_source_module_analytical_characterization ${ICTS_MODULE_ANALYTICAL_CHARACTERIZATION_SRC})

target_link_libraries(
  icts_source_module_analytical_characterization
  PUBLIC
  icts_source_database_characterization
  icts_source_database_routing
  PRIVATE
  icts_source_module_characterization
)

target_include_directories(
  icts_source_module_analytical_characterization
  PUBLIC
  ${ICTS_MODULE}
  ${ICTS_MODULE_ANALYTICAL_CHARACTERIZATION}
)

if(DEBUG_ICTS_SOURCE_MODULE_ANALYTICAL_CHARACTERIZATION)
  if(MSVC)
    target_compile_options(icts_source_module_analytical_characterization PRIVATE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_module_analytical_characterization PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
