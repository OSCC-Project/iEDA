add_library(imp-module 
            ${IMP_MODULE}/SA.cc
            ${IMP_MODULE}/NetList.cc)

target_link_libraries(imp-module
                      PRIVATE
                      imp-solver)
target_include_directories(imp-module
                           PUBLIC
                           ${IMP_MODULE})
add_library(imp-netlist ${IMP_MODULE}/NetList.cc)
target_link_libraries(imp-netlist
                      PRIVATE
                      imp-solver)
target_include_directories(imp-netlist
                           PUBLIC
                           ${IMP_MODULE})
set(IMP_REPRESENTATION ${IMP_MODULE}/representation)
add_subdirectory(${IMP_REPRESENTATION})