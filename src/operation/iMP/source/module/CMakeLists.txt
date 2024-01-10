set(IMP_OPERATOR ${IMP_MODULE}/operator)
add_subdirectory(${IMP_OPERATOR})
set(IMP_PARTITION ${IMP_MODULE}/partition)
add_subdirectory(${IMP_PARTITION})
# add_library(imp-netlist ${IMP_MODULE}/NetList.cc)
# target_link_libraries(imp-netlist
#                       PRIVATE
#                       imp-solver
#                       imp-structure)
# target_include_directories(imp-netlist
#                            PUBLIC
#                            ${IMP_MODULE})
# set(IMP_REPRESENTATION ${IMP_MODULE}/representation)
# set(IMP_RANDOMSEARCH ${IMP_MODULE}/randomsearch)
# add_subdirectory(${IMP_REPRESENTATION})
# add_subdirectory(${IMP_RANDOMSEARCH})