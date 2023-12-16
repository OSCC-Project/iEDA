add_library(imp-heuristic
           ${IMP_HEURISTIC}/SA.cc
           ${IMP_HEURISTIC}/SAPlacement.tpp
           ${IMP_HEURISTIC}/SAPlacementPlot.tpp)

target_link_libraries(imp-heuristic
                      PRIVATE
                      imp-representation
                      imp-solver
                      )
target_include_directories(imp-heuristic
                          PUBLIC 
                           ${IMP_HEURISTIC})