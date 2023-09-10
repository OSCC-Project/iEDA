if(NOT BUILD_STATIC_LIB)

SET(CMAKE_BUILD_TYPE "Debug")

AUX_SOURCE_DIRECTORY(./ SRC)

pybind11_add_module(ista_cpp MODULE ${SRC})
target_link_libraries(ista_cpp PRIVATE sta ista-engine)
set_target_properties(ista_cpp PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

endif()