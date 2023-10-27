add_executable(iMP-test ${IMP_TEST}/Test.cc)
target_link_libraries(iMP-test 
   PRIVATE
   imp_source
   # imp-plot
   )
add_executable(test_hypergraph ${IMP_TEST}/hypergraph.cc)
target_link_libraries(test_hypergraph
PRIVATE
imp-structure) 