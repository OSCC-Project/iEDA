add_library(icts_api ${ICTS_API}/CTSAPI.cc)

target_include_directories(icts_api PUBLIC ${ICTS_API})

target_link_libraries(
  icts_api
  PRIVATE
  icts_source
  icts_api_external_libs
)

if(DEBUG_ICTS_API)
  if(MSVC)
    target_compile_options(icts_api PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_api PRIVATE /DEBUG)
  else()
    target_compile_options(icts_api PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
