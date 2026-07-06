set(IRCX_ITF_DATA_SRC
  Layer.cpp
  ProcessCorner.cpp
  VariationParams.cpp
)

add_library(itf_data ${IRCX_ITF_DATA_SRC})

target_link_libraries(itf_data
  PUBLIC
    itf
)

target_include_directories(itf_data
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
