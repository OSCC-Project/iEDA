
add_library(imp-netlist ${IMP_MODULE}/NetList.cc ${IMP_MODULE}/Evaluator.cc)
target_link_libraries(imp-netlist
                      PRIVATE
                      imp-solver)
target_include_directories(imp-netlist
                           PUBLIC
                           ${IMP_MODULE})
set(IMP_REPRESENTATION ${IMP_MODULE}/representation)
set(IMP_HEURISTIC ${IMP_MODULE}/heuristic)
add_subdirectory(${IMP_REPRESENTATION})
add_subdirectory(${IMP_HEURISTIC})