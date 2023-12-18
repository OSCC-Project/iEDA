add_library(imp-representation 
           ${IMP_REPRESENTATION}/SeqPair.tpp
           )
           
set_target_properties(imp-representation PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(imp-representation
                          PUBLIC 
                           ${IMP_REPRESENTATION})   