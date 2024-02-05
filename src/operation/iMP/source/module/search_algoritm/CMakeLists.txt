add_library(imp-search-alg INTERFACE )
target_include_directories(imp-search-alg
                          INTERFACE 
                           ${IMP_SEARCH})

add_executable(iMP-example-sa ${IMP_SEARCH}/SP_SAPlace_example.cc)
target_link_libraries(iMP-example-sa 
   PRIVATE
   imp-representation
   imp-search-alg
   imp-formula
   imp-plot
   )
# target_include_directories(iMP-sp-test
# )