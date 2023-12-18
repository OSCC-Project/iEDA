add_library(imp-module 
            ${IMP_MODULE}/SA.cc
            ${IMP_MODULE}/Evaluator.cc
            ${IMP_MODULE}/NetList.cc)

target_link_libraries(imp-module
                      PUBLIC
                      imp-solver)
target_include_directories(imp-module
                           PUBLIC
                           ${IMP_MODULE})