
add_library(imp-netlist ${IMP_MODULE}/NetList.cc)
target_link_libraries(imp-netlist
                      PRIVATE
                      imp-solver)
target_include_directories(imp-netlist
                           PUBLIC
                           ${IMP_MODULE})
set(IMP_REPRESENTATION ${IMP_MODULE}/representation)
set(IMP_RANDOMSEARCH ${IMP_MODULE}/randomsearch)
add_subdirectory(${IMP_REPRESENTATION})
add_subdirectory(${IMP_RANDOMSEARCH})