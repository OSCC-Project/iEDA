set(ICTS_DATABASE ${ICTS_SOURCE}/database)
set(ICTS_FLOW ${ICTS_SOURCE}/flow)
set(ICTS_MODULE ${ICTS_SOURCE}/module)
set(ICTS_UTILS ${ICTS_SOURCE}/utils)

add_subdirectory(${ICTS_DATABASE})
add_subdirectory(${ICTS_UTILS})
add_subdirectory(${ICTS_MODULE})
add_subdirectory(${ICTS_FLOW})

add_library(icts_source INTERFACE)

target_link_libraries(
  icts_source
  INTERFACE
  icts_source_flow
  icts_source_database
  icts_source_utils
)

target_include_directories(icts_source INTERFACE ${ICTS_SOURCE})

if(DEBUG_ICTS_SOURCE)
  if(MSVC)
    target_compile_options(icts_source INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
