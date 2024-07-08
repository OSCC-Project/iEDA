if(DEBUG_ITO_VG_BUFFER)
  message(STATUS "TO: DEBUG_ITO_VG_BUFFER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "TO: RELEASE_ITO_VG_BUFFER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_vg_buffer
BufferedOption.cpp
VGBuffer.cpp
)

target_link_libraries(ito_vg_buffer
    PUBLIC
    ito_utility
    ito_timing_engine
    ito_config
    ito_data_manager
    ito_tree_build
)

target_include_directories(ito_vg_buffer
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)

