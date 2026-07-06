set(IRCX_ITF_BUILDER_SRC
  ItfBuilder.cpp
  ItfRead.cpp
)

add_library(itf_builder ${IRCX_ITF_BUILDER_SRC})

target_link_libraries(itf_builder
  PUBLIC
    itf
    itf_data
)

target_include_directories(itf_builder
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
