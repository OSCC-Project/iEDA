if(NOT BUILD_STATIC_LIB)

SET(CMAKE_BUILD_TYPE "Debug")

AUX_SOURCE_DIRECTORY(./ SRC)

pybind11_add_module(ipower_cpp MODULE ${SRC})
target_link_libraries(ipower_cpp PRIVATE vcd_wrapper log power verilog-vcd-parser utility usage ista-engine core calc_power plot_power sta)
# set_target_properties(ipower_cpp PROPERTIES 
#   LIBRARY_OUTPUT_DIRECTORY /home/shaozheqing/iEDA/src/operation/iPW/test)

endif()