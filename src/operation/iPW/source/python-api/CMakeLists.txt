if(NOT BUILD_STATIC_LIB)

set(CMAKE_BUILD_TYPE "Debug")

aux_source_directory(./ SRC)

pybind11_add_module(ipower_cpp MODULE ${SRC})
target_link_libraries(ipower_cpp PRIVATE vcd_wrapper log power verilog-vcd-parser utility usage ista-engine core calc_power plot_power sta)
set_target_properties(ipower_cpp PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

endif()